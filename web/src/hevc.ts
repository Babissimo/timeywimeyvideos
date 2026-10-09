/**
 * The frame rate an HEVC stream declares in its parameter sets, as ffmpeg's decoder takes it:
 * the video parameter set's (VPS) timing, else the timing in the sequence parameter set's
 * (SPS) VUI. Field names follow the H.265 standard's syntax tables (7.3.2.1, 7.3.2.2.1,
 * 7.3.3, 7.3.4, 7.3.7 and E.2.1).
 */
import { Bits, unescape } from "./avc";
import type { Rate } from "./types";

const VPS = 32;
const SPS = 33;

/** Skip a profile_tier_level with its general profile and `subLayers` sub-layers above the first. */
function skipProfileTierLevel(bits: Bits, subLayers: number): void {
  bits.skip(96);  // general profile, tier, compatibility and constraint flags, and level
  const present: [number, number][] = [];
  for (let i = 0; i < subLayers; i++) present.push([bits.u(1), bits.u(1)]);
  if (subLayers > 0) bits.skip(2 * (8 - subLayers));
  for (const [profile, level] of present) bits.skip(88 * profile + 8 * level);
}

function skipScalingListData(bits: Bits): void {
  for (let size = 0; size < 4; size++) {
    for (let matrix = 0; matrix < 6; matrix += size === 3 ? 3 : 1) {
      if (!bits.u(1)) {  // scaling_list_pred_mode_flag
        bits.ue();
        continue;
      }
      if (size > 1) bits.se();  // scaling_list_dc_coef_minus8
      for (let i = Math.min(64, 1 << (4 + (size << 1))); i > 0; i--) bits.se();
    }
  }
}

/**
 * Skip an SPS's short-term reference picture sets. A set predicted from the one before
 * flags which of that one's pictures it keeps, so each set's pictures are followed: their
 * POC differences from the current picture, those before it then those after (7.4.8).
 */
function skipShortTermRefPicSets(bits: Bits, count: number): void {
  let before: number[] = [];
  let after: number[] = [];
  for (let i = 0; i < count; i++) {
    const nextBefore: number[] = [];
    const nextAfter: number[] = [];
    if (i > 0 && bits.u(1)) {  // inter_ref_pic_set_prediction_flag
      const sign = bits.u(1);
      const delta = (bits.ue() + 1) * (sign ? -1 : 1);
      const n = before.length + after.length;
      // used_by_curr_pic_flag, and use_delta_flag where that is 0, for each picture of the
      // set before and for the picture delta away.
      const used = Array.from({ length: n + 1 }, () => bits.u(1) === 1 || bits.u(1) === 1);
      const keep = (list: number[], d: number, j: number, side: number) => {
        if (used[j] && d * side > 0) list.push(d);
      };
      for (let j = after.length - 1; j >= 0; j--) keep(nextBefore, after[j]! + delta, before.length + j, -1);
      keep(nextBefore, delta, n, -1);
      for (let j = 0; j < before.length; j++) keep(nextBefore, before[j]! + delta, j, -1);
      for (let j = before.length - 1; j >= 0; j--) keep(nextAfter, before[j]! + delta, j, 1);
      keep(nextAfter, delta, n, 1);
      for (let j = 0; j < after.length; j++) keep(nextAfter, after[j]! + delta, before.length + j, 1);
    } else {
      const negatives = bits.ue();
      const positives = bits.ue();
      for (let j = 0, poc = 0; j < negatives; j++) {
        poc -= bits.ue() + 1;
        bits.skip(1);  // used_by_curr_pic_s0_flag
        nextBefore.push(poc);
      }
      for (let j = 0, poc = 0; j < positives; j++) {
        poc += bits.ue() + 1;
        bits.skip(1);  // used_by_curr_pic_s1_flag
        nextAfter.push(poc);
      }
    }
    [before, after] = [nextBefore, nextAfter];
  }
}

/** A VPS payload's vps_timing_info as [num_units_in_tick, time_scale], or null without one. */
function vpsTiming(rbsp: Uint8Array): [number, number] | null {
  const bits = new Bits(rbsp);
  bits.skip(12);  // vps_video_parameter_set_id, base layer flags, vps_max_layers_minus1
  const subLayers = bits.u(3);
  bits.skip(17);  // vps_temporal_id_nesting_flag, vps_reserved_0xffff_16bits
  skipProfileTierLevel(bits, subLayers);
  for (let i = bits.u(1) ? 0 : subLayers; i <= subLayers; i++) {  // sub_layer_ordering_info
    bits.ue();
    bits.ue();
    bits.ue();
  }
  const layers = bits.u(6) + 1;  // vps_max_layer_id
  bits.skip(bits.ue() * layers);  // layer_id_included_flag for each layer set after the first
  return bits.u(1) ? [bits.u(32), bits.u(32)] : null;
}

/** An SPS payload's VUI timing as [num_units_in_tick, time_scale], or null without one. */
function spsTiming(rbsp: Uint8Array): [number, number] | null {
  const bits = new Bits(rbsp);
  bits.skip(4);  // sps_video_parameter_set_id
  const subLayers = bits.u(3);
  bits.skip(1);  // sps_temporal_id_nesting_flag
  skipProfileTierLevel(bits, subLayers);
  bits.ue();  // sps_seq_parameter_set_id
  if (bits.ue() === 3) bits.skip(1);  // chroma_format_idc, separate_colour_plane_flag
  bits.ue();
  bits.ue();  // picture size
  if (bits.u(1)) for (let i = 0; i < 4; i++) bits.ue();  // conformance window
  bits.ue();
  bits.ue();  // bit depths
  const pocBits = bits.ue() + 4;
  for (let i = bits.u(1) ? 0 : subLayers; i <= subLayers; i++) {  // sub_layer_ordering_info
    bits.ue();
    bits.ue();
    bits.ue();
  }
  for (let i = 0; i < 6; i++) bits.ue();  // coding and transform block sizes and depths
  if (bits.u(1) && bits.u(1)) skipScalingListData(bits);
  bits.skip(2);  // amp_enabled_flag, sample_adaptive_offset_enabled_flag
  if (bits.u(1)) {  // pcm_enabled_flag
    bits.skip(8);
    bits.ue();
    bits.ue();
    bits.skip(1);
  }
  skipShortTermRefPicSets(bits, bits.ue());
  if (bits.u(1)) bits.skip(bits.ue() * (pocBits + 1));  // long-term reference pictures
  bits.skip(2);  // sps_temporal_mvp_enabled_flag, strong_intra_smoothing_enabled_flag
  if (!bits.u(1)) return null;  // vui_parameters_present_flag
  if (bits.u(1) && bits.u(8) === 255) bits.skip(32);  // aspect ratio, and an explicit one
  if (bits.u(1)) bits.skip(1);  // overscan
  if (bits.u(1)) {  // video signal type
    bits.skip(4);
    if (bits.u(1)) bits.skip(24);
  }
  if (bits.u(1)) {  // chroma location
    bits.ue();
    bits.ue();
  }
  bits.skip(3);  // neutral_chroma_indication_flag, field_seq_flag, frame_field_info_present_flag
  if (bits.u(1)) for (let i = 0; i < 4; i++) bits.ue();  // default display window
  return bits.u(1) ? [bits.u(32), bits.u(32)] : null;
}

/**
 * The frame rate an HEVCDecoderConfigurationRecord's first SPS, with the VPS it refers to,
 * declares: time_scale over num_units_in_tick, unreduced; null if it declares none.
 */
export function hevcFrameRate(record: Uint8Array): Rate | null {
  const sets = new Map<number, Uint8Array[]>();
  let at = 23;
  for (let i = 0; i < record[22]!; i++) {
    const type = record[at]! & 0x3f;
    const count = (record[at + 1]! << 8) | record[at + 2]!;
    at += 3;
    for (let j = 0; j < count; j++) {
      const length = (record[at]! << 8) | record[at + 1]!;
      sets.set(type, [...sets.get(type) ?? [], record.subarray(at + 2, at + 2 + length)]);
      at += 2 + length;
    }
  }
  const sps = sets.get(SPS)?.[0];
  if (!sps) return null;
  // Past each two-byte NAL unit header, a parameter set starts with its VPS's ID.
  const vps = sets.get(VPS)?.find((nal) => nal[2]! >> 4 === sps[2]! >> 4);
  const timing = vps && vpsTiming(unescape(vps.subarray(2)));
  const [units, scale] = timing ?? spsTiming(unescape(sps.subarray(2))) ?? [0, 0];
  return units > 0 && scale > 0 ? { num: scale, den: units } : null;
}

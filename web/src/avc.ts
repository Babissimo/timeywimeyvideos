/**
 * Marking an H.264 stream's colour in its sequence parameter sets (SPS), for encoders that
 * leave it out. A decoder that finds no colour description in the stream guesses one, by the
 * picture's size, and a guess of BT.601 for a picture marked BT.709 in its MP4 shifts every
 * colour. Field names follow the H.264 standard's syntax tables (7.3.2.1.1 and E.1.1).
 */

/** The colour description a VUI carries, as H.264's Table E-3, E-4 and E-5 number them. */
export interface Colour { primaries: number; transfer: number; matrix: number; fullRange: boolean }

/** BT.709 at limited range. */
export const BT709: Colour = { primaries: 1, transfer: 1, matrix: 1, fullRange: false };

// The profiles whose SPS has chroma format, bit depth and scaling matrix fields.
const HIGH_PROFILES = new Set([100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135]);

/** An SPS's payload without the bytes that keep start codes out of it. */
function unescape(nal: Uint8Array): Uint8Array {
  const out: number[] = [];
  let zeros = 0;
  for (const byte of nal) {
    if (zeros >= 2 && byte === 3) {
      zeros = 0;
      continue;
    }
    out.push(byte);
    zeros = byte === 0 ? zeros + 1 : 0;
  }
  return Uint8Array.from(out);
}

/** A payload with those bytes put back. */
function escape(rbsp: number[]): number[] {
  const out: number[] = [];
  let zeros = 0;
  for (const byte of rbsp) {
    if (zeros >= 2 && byte <= 3) {
      out.push(3);
      zeros = 0;
    }
    out.push(byte);
    zeros = byte === 0 ? zeros + 1 : 0;
  }
  return out;
}

class Bits {
  at = 0;
  constructor(readonly data: Uint8Array) {}

  bit(): number {
    const byte = this.data[this.at >> 3];
    if (byte === undefined) throw new RangeError("the SPS ends early");
    return (byte >> (7 - (this.at++ & 7))) & 1;
  }

  u(n: number): number {
    let value = 0;
    for (let i = 0; i < n; i++) value = value * 2 + this.bit();
    return value;
  }

  ue(): number {
    let zeros = 0;
    while (this.bit() === 0) zeros++;
    return 2 ** zeros - 1 + this.u(zeros);
  }

  se(): number {
    const k = this.ue();
    return k % 2 ? (k + 1) / 2 : -k / 2;
  }
}

/** Skip a scaling_list of n coefficients. */
function skipScalingList(bits: Bits, n: number): void {
  let last = 8, next = 8;
  for (let j = 0; j < n; j++) {
    if (next !== 0) next = (last + bits.se() + 256) % 256;
    last = next === 0 ? last : next;
  }
}

/** Where an SPS payload's VUI starts, read up to vui_parameters_present_flag. */
interface Layout {
  /** The bit holding vui_parameters_present_flag. */
  vui: number;
  /** With a VUI: the bit holding video_signal_type_present_flag, and the first after its fields. */
  signal?: [number, number];
  /** The rbsp_stop_one_bit. */
  stop: number;
}

function layout(rbsp: Uint8Array): Layout {
  const bits = new Bits(rbsp);
  const profile = bits.u(8);
  bits.u(16);  // constraint flags and level_idc
  bits.ue();   // seq_parameter_set_id
  if (HIGH_PROFILES.has(profile)) {
    const chromaFormat = bits.ue();
    if (chromaFormat === 3) bits.u(1);  // separate_colour_plane_flag
    bits.ue();
    bits.ue();   // bit depths
    bits.u(1);   // qpprime_y_zero_transform_bypass_flag
    if (bits.u(1)) {  // seq_scaling_matrix_present_flag
      for (let i = 0; i < (chromaFormat === 3 ? 12 : 8); i++)
        if (bits.u(1)) skipScalingList(bits, i < 6 ? 16 : 64);
    }
  }
  bits.ue();  // log2_max_frame_num_minus4
  const pocType = bits.ue();
  if (pocType === 0) {
    bits.ue();
  } else if (pocType === 1) {
    bits.u(1);
    bits.se();
    bits.se();
    const cycle = bits.ue();
    for (let i = 0; i < cycle; i++) bits.se();
  }
  bits.ue();  // max_num_ref_frames
  bits.u(1);  // gaps_in_frame_num_value_allowed_flag
  bits.ue();
  bits.ue();  // picture size in macroblocks
  if (!bits.u(1)) bits.u(1);  // frame_mbs_only_flag, mb_adaptive_frame_field_flag
  bits.u(1);  // direct_8x8_inference_flag
  if (bits.u(1)) for (let i = 0; i < 4; i++) bits.ue();  // frame cropping
  const vui = bits.at;
  let stop = rbsp.length * 8 - 1;
  while (stop >= 0 && !((rbsp[stop >> 3] >> (7 - (stop & 7))) & 1)) stop--;
  if (!bits.u(1)) return { vui, stop };
  if (bits.u(1) && bits.u(8) === 255) bits.u(32);  // aspect ratio, and an explicit one
  if (bits.u(1)) bits.u(1);  // overscan
  const signal = bits.at;
  if (bits.u(1)) {
    bits.u(4);  // video_format, video_full_range_flag
    if (bits.u(1)) bits.u(24);
  }
  return { vui, signal: [signal, bits.at], stop };
}

/** Writes bits into bytes. */
class Writer {
  readonly bytes: number[] = [];
  private n = 0;

  bit(b: number): void {
    if (this.n % 8 === 0) this.bytes.push(0);
    if (b) this.bytes[this.bytes.length - 1] |= 0x80 >> (this.n % 8);
    this.n++;
  }

  u(n: number, value: number): void {
    for (let i = n - 1; i >= 0; i--) this.bit(Math.floor(value / 2 ** i) % 2);
  }

  copy(from: Bits, to: number): void {
    while (from.at < to) this.bit(from.bit());
  }

  /** rbsp_trailing_bits. */
  end(): void {
    this.bit(1);
    while (this.n % 8) this.bit(0);
  }
}

function signalType(out: Writer, colour: Colour): void {
  out.u(1, 1);  // video_signal_type_present_flag
  out.u(3, 5);  // video_format: unspecified
  out.u(1, Number(colour.fullRange));
  out.u(1, 1);  // colour_description_present_flag
  out.u(8, colour.primaries);
  out.u(8, colour.transfer);
  out.u(8, colour.matrix);
}

/** An SPS NAL unit with its VUI's video signal type set to this colour, the rest as it was. */
export function markSps(nal: Uint8Array, colour: Colour): Uint8Array {
  const rbsp = unescape(nal.subarray(1));
  const { vui, signal, stop } = layout(rbsp);
  const bits = new Bits(rbsp);
  const out = new Writer();
  if (signal) {
    out.copy(bits, signal[0]);
    signalType(out, colour);
    bits.at = signal[1];
    out.copy(bits, stop);
  } else {
    out.copy(bits, vui);
    out.u(1, 1);  // vui_parameters_present_flag
    out.u(2, 0);  // no aspect ratio or overscan
    signalType(out, colour);
    // No chroma location, timing, HRD, picture structure or bitstream restrictions.
    out.u(6, 0);
  }
  out.end();
  return Uint8Array.from([nal[0], ...escape(out.bytes)]);
}

/** The colour an SPS NAL unit's VUI describes, or null if it describes none. */
export function spsColour(nal: Uint8Array): Colour | null {
  const rbsp = unescape(nal.subarray(1));
  const { signal } = layout(rbsp);
  if (!signal) return null;
  const bits = new Bits(rbsp);
  bits.at = signal[0];
  if (!bits.u(1)) return null;
  bits.u(3);
  const fullRange = Boolean(bits.u(1));
  if (!bits.u(1)) return null;
  return { primaries: bits.u(8), transfer: bits.u(8), matrix: bits.u(8), fullRange };
}

/**
 * An AVCDecoderConfigurationRecord (an MP4's avcC box, a WebCodecs H.264 description) with
 * each SPS marked with this colour; the rest is as it was.
 */
export function markColour(record: Uint8Array, colour: Colour): Uint8Array {
  const out: number[] = [...record.subarray(0, 5)];
  let at = 5;
  const sets = record[at++] & 0x1f;
  out.push(0xe0 | sets);
  for (let i = 0; i < sets; i++) {
    const length = (record[at] << 8) | record[at + 1];
    const sps = markSps(record.subarray(at + 2, at + 2 + length), colour);
    out.push(sps.length >> 8, sps.length & 0xff, ...sps);
    at += 2 + length;
  }
  out.push(...record.subarray(at));  // the picture parameter sets, and any extension
  return Uint8Array.from(out);
}

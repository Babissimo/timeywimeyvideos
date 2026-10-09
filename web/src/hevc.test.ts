import { expect, test } from "vitest";
import { hevcFrameRate } from "./hevc";

const hex = (text: string) => Uint8Array.from(text.match(/../g)!, (h) => parseInt(h, 16));

/** An HEVCDecoderConfigurationRecord holding these NAL units, each in an array of its own. */
function record(...nals: string[]): Uint8Array {
  const out = [1, ...Array<number>(21).fill(0), nals.length];
  for (const nal of nals.map(hex)) {
    out.push(0x80 | (nal[0]! >> 1), 0, 1, nal.length >> 8, nal.length & 0xff, ...nal);
  }
  return Uint8Array.from(out);
}

// Video and sequence parameter sets as encoders wrote them, and as made here to try rarer
// syntax, with the rate ffmpeg's trace_headers reads from their timing.
const X265_VPS = "40010c01ffff01600000030090000003000003003c959809";
const CASES: [string, string[], { num: number; den: number } | null][] = [
  ["x265 at 30 fps", [X265_VPS, "42010101600000030090000003000003003ca00a080f165959a4932bc05a020000030002000003003c10"],
   { num: 30, den: 1 }],
  ["x265 at 60 fps with B-pyramids", [
    "40010c01ffff01600000030090000003000003003f998a0240",
    "42010101600000030090000003000003003fa00a080f165998a924caf0168080000003008000001e04",
  ], { num: 60, den: 1 }],
  ["x265 without VUI timing", [X265_VPS, "42010101600000030090000003000003003ca00a080f165959a4932bc05a0020"],
   null],
  // Default scaling lists and four explicit reference picture sets, but no timing.
  ["Apple's VideoToolbox", [
    "40010c01ffff016000000300b0000003000003005d170240",
    "420101016000000300b0000003000003005da00a080f136205ee45914bff2e7f13fac05a8101010040",
  ], null],
  // Two sub-layers, scaling lists given in full, PCM, six reference picture sets (four
  // predicted, one from a set smaller than its own reference), two long-term reference
  // pictures and every optional VUI field before the timing.
  ["made: an SPS with the rarer syntax", [
    "40010c05ffff01600000030090000003000003005dd00001600000030090000003000003005a5a95cae5709da4",
    "42010501600000030090000003000003005dd00001600000030090000003000003005a5aa00a080f1fe515e4912e57442b"
    + "a215d10a52ba215d10ae885295d10ae8857442932ba215d10ae8857442ba215d10ae8857442ba215d10ae8857442ba65"
    + "7442ba215d10ae8857442ba215d10ae8857442ba215d10ae88574cae8857442ba215d10ae8857442ba215d10ae885744"
    + "2ba215d10aea72ba215d10ae8857442ba215d10ae8857442ba215d10ae8857442ba29cae8857442ba215d10ae8857442"
    + "ba215d10ae8857442ba215d10ae8a72ba215d10ae8857442ba215d10ae8857442ba215d10ae8857442ba2672ba215d10a"
    + "e8857442ba215d10ae8857442ba215d10ae8857442ba6ef43b5afe85b2932bb7b0584bffc0010000fd404040690ddc00"
    + "000fa40003a9802",
  ], { num: 60000, den: 1001 }],
  // The VPS's timing, 25 fps, wins over the SPS's 30.
  ["made: timing in the VPS", [
    "40010c01ffff01600000030090000003000003005d95c276c0000003004000000654",
    "42010101600000030090000003000003005da00a080f1fe515e4912377a36b5fd07ff80020001fa808080d21bb800000"
    + "03008000000f04",
  ], { num: 25, den: 1 }],
];

test.each(CASES)("hevcFrameRate reads the rate declared: %s", (_, nals, rate) => {
  expect(hevcFrameRate(record(...nals))).toEqual(rate);
});

test("hevcFrameRate finds none without an SPS", () => {
  expect(hevcFrameRate(record(X265_VPS))).toBeNull();
});

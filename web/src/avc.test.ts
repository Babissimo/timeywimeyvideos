import { describe, expect, test } from "vitest";
import { BT709, markColour, markSps, spsColour, type Colour } from "./avc";

const hex = (text: string) => Uint8Array.from(text.match(/../g)!, (h) => parseInt(h, 16));

// Sequence parameter sets as encoders wrote them.
const SPS = {
  // Chrome's hardware encoder with a quantizer: a VUI, but no colour.
  noColour: "2764001eac1316240a02ff97000c0c0017700005dc17bdf07c22118980",
  // Chrome's hardware encoder with a bitrate: BT.709.
  bt709: "2764001eac1316c0a02ff97016a020202606000bb80002ee0bdef83e1108dc",
  // Chrome's software encoder: BT.709 primaries and matrix, sRGB transfer.
  srgb: "67640c0aac18d1ab1124d40434041e1108d4",
  // x264, with bytes escaped to keep start codes out.
  escaped: "67640028acd9406c0227b9610000030001000003003c0f183196",
  // x264's, cut short before its VUI.
  noVui: "6764000aacd94479",
};

const BT601: Colour = { primaries: 6, transfer: 6, matrix: 6, fullRange: true };

/** The payload's bits, without escapes, as a string of 0s and 1s up to the stop bit. */
function bits(nal: Uint8Array): string {
  const payload: number[] = [];
  let zeros = 0;
  for (const byte of nal.subarray(1)) {
    if (zeros >= 2 && byte === 3) {
      zeros = 0;
      continue;
    }
    payload.push(byte);
    zeros = byte === 0 ? zeros + 1 : 0;
  }
  const all = payload.map((b) => b.toString(2).padStart(8, "0")).join("");
  return all.slice(0, all.lastIndexOf("1"));
}

describe("markSps", () => {
  test("reads the colour an SPS describes", () => {
    expect(spsColour(hex(SPS.noColour))).toBeNull();
    expect(spsColour(hex(SPS.noVui))).toBeNull();
    expect(spsColour(hex(SPS.bt709))).toEqual(BT709);
    expect(spsColour(hex(SPS.srgb))).toEqual({ ...BT709, transfer: 13 });
    expect(spsColour(hex(SPS.escaped))).toBeNull();
  });

  test("leaves an SPS as it was when it already describes the colour", () => {
    expect(markSps(hex(SPS.bt709), BT709)).toEqual(hex(SPS.bt709));
  });

  for (const [name, sps] of Object.entries(SPS)) {
    test(`marks the colour, keeping the rest: ${name}`, () => {
      for (const colour of [BT709, BT601]) {
        const marked = markSps(hex(sps), colour);
        expect(marked[0]).toBe(hex(sps)[0]);
        expect(spsColour(marked)).toEqual(colour);
        expect(markSps(marked, colour)).toEqual(marked);
        // Escaped as it must be: no start code inside.
        expect(Buffer.from(marked).indexOf(Buffer.from([0, 0, 1]))).toBe(-1);
        expect(Buffer.from(marked).indexOf(Buffer.from([0, 0, 0]))).toBe(-1);
      }
    });
  }

  test("changes only the video signal type", () => {
    // Up to video_signal_type_present_flag, then from the first bit after its fields.
    const cases: [string, number, number][] = [
      [SPS.noColour, 97, 98], [SPS.bt709, 93, 123], [SPS.escaped, 85, 86], [SPS.srgb, 72, 102],
    ];
    for (const [sps, from, to] of cases) {
      const before = bits(hex(sps)), after = bits(markSps(hex(sps), BT601));
      expect(after.slice(0, from)).toBe(before.slice(0, from));
      expect(after.slice(from, from + 30)).toBe("1" + "101" + "1" + "1" + "00000110".repeat(3));
      expect(after.slice(from + 30)).toBe(before.slice(to));
    }
  });

  test("adds a VUI holding only the colour where there is none", () => {
    const before = bits(hex(SPS.noVui)), after = bits(markSps(hex(SPS.noVui), BT709));
    const vui = before.length - 1;  // its last bit is vui_parameters_present_flag, 0
    expect(after.slice(0, vui)).toBe(before.slice(0, vui));
    expect(after.slice(vui)).toBe("1" + "00" + "1" + "101" + "0" + "1" + "00000001".repeat(3)
                                  + "000000");
  });
});

test("markColour marks each SPS of a decoder configuration record, and keeps the rest", () => {
  const pps = hex("28ef1f2c"), tail = hex("fdf8f800");
  const sps = hex(SPS.noColour);
  const record = Uint8Array.from([1, 0x64, 0, 0x1e, 0xff, 0xe1, 0, sps.length, ...sps,
                                  1, 0, pps.length, ...pps, ...tail]);
  const marked = markColour(record, BT709);
  const spsOut = markSps(sps, BT709);
  expect(marked).toEqual(Uint8Array.from([1, 0x64, 0, 0x1e, 0xff, 0xe1, 0, spsOut.length,
                                          ...spsOut, 1, 0, pps.length, ...pps, ...tail]));
});

import { expect, test } from "vitest";
import { readOptions, type OptionValues } from "./options";
import { outputName } from "./render";

const named = (name: string, values: OptionValues, preview = false) =>
  outputName(name, readOptions(values), preview);

test("names a render after its clip and the options that differ from the defaults", () => {
  expect(named("clip.mp4", {})).toBe("clip_rotate_45deg.mp4");
  expect(named("clip.mp4", {}, true)).toBe("clip_rotate_45deg_preview.mp4");
  expect(named("a.b.mov", { angle: "-30.5", slice: "shear", inside: "1" }))
    .toBe("a.b_shear_-30.5deg_inside.mp4");
  expect(named("my clip's", { angle: "80", inside: "1", motion: "perpendicular" }))
    .toBe("my clip's_rotate_80deg_inside_perpendicular.mp4");
  expect(named(".hidden.mp4", { loop: "1", loop_fade: "0.25", loop_sides: "1", motion: "time",
                                loop_side_fade: "12" }))
    .toBe(".hidden_rotate_45deg_time_loop_loopfade0.25s_sides_sidefade12px.mp4");
  expect(named("clip.mp4", { noise: "4" })).toBe("clip_rotate_45deg_noise4.mp4");
  expect(named("clip.mp4", { noise: "2.5", noise_size: "32", noise_speed: "0.5",
                             noise_direction: "perpendicular", noise_seed: "7" }))
    .toBe("clip_rotate_45deg_noise2.5_noisesize32_noisespeed0.5_noiseperpendicular_"
          + "noiseseed7.mp4");
  expect(named("clip.mp4", { start: "1.5", duration: "2", scale: "0.5", fps: "30000/1001" }))
    .toBe("clip_rotate_45deg_from1.5s_for2s_scale0.5_30000over1001fps.mp4");
  expect(named("clip.mp4", { fps: "29.97" })).toBe("clip_rotate_45deg_2997over100fps.mp4");
});

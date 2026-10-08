/** GLSL ES 3.00 sources for slicing a volume. */

/** One triangle covering the viewport, corners (-1, -1), (3, -1) and (-1, 3). */
export const COVER = `#version 300 es
void main() {
  gl_Position = vec4(float(gl_VertexID & 1) * 4.0 - 1.0, float(gl_VertexID >> 1) * 4.0 - 1.0,
                     0.0, 1.0);
}`;

const HEADER = `#version 300 es
precision highp float;
precision highp int;
`;

const VOXEL = `
uniform highp sampler2DArray volume;  // texel (x, y, layer) is voxel (t = layer, y, x)

// Voxel (t, y, x), in levels from 0 to 255.
ivec4 voxel(int t, int y, int x) {
  return ivec4(round(texelFetch(volume, ivec3(x, y, t), 0) * 255.0));
}
`;

const BLACK = "uvec4(0u, 0u, 0u, 255u)";

// timeslice blends voxels a = (t0, x0), b = (t0, x1), c = (t1, x0) and d = (t1, x1) in
// float64 as v = (a (1 - fx) + b fx) (1 - ft) + (c (1 - fx) + d fx) ft, and rounds it as
// min(255, int(v + 0.5)). Here v = a + fx (b - a) + ft (c - a) + fx ft (a - b - c + d) is
// summed exactly in integers, with fx, ft and fx ft truncated to whole numbers of 2^-36, in
// three 12-bit parts, w[0] the most significant; the floored carries floor the whole sum. So
// the level differs from timeslice's only where v is within about 1e-8 of halfway between
// two. In floats the compiler may reorder the sums and round differently; in integers the
// order makes no difference.
const BLEND = `
uvec4 blend(ivec4 a, ivec4 b, ivec4 c, ivec4 d, ivec3 w[3]) {
  ivec4 e1 = b - a, e2 = c - a, e3 = a - b - c + d;
  ivec4 sum = ivec4(0);
  for (int i = 2; i >= 0; i--) sum = (sum >> 12) + w[i].x * e1 + w[i].y * e2 + w[i].z * e3;
  return uvec4(clamp(a + ((sum + 2048) >> 12), 0, 255));  // sum is v - a in units of 2^-12
}
`;

/** A flat slice: output column j reads the voxels and weights worked out for it. */
export const FLAT = `${HEADER}${VOXEL}${BLEND}
// Per column: voxels t0, t1, x0, x1 in row 0, t0 -1 outside the video, then a part of
// each of the blend's weights in each of rows 1 to 3.
uniform highp isampler2D columns;
uniform bool nearest;  // copy voxel (t0, x0), already the closest
out uvec4 colour;

void main() {
  int j = int(gl_FragCoord.x), row = int(gl_FragCoord.y);
  ivec4 at = texelFetch(columns, ivec2(j, 0), 0);
  if (at.x < 0) {
    colour = ${BLACK};
  } else if (nearest) {
    colour = uvec4(voxel(at.x, row, at.z));
  } else {
    ivec3 w[3];
    for (int i = 0; i < 3; i++) w[i] = texelFetch(columns, ivec2(j, i + 1), 0).xyz;
    colour = blend(voxel(at.x, row, at.z), voxel(at.x, row, at.w),
                   voxel(at.y, row, at.z), voxel(at.y, row, at.w), w);
  }
}`;

/** The slice scaled to the canvas, nearest pixel, its first row at the top. */
export const SHOW = `${HEADER}
uniform highp usampler2D slice;
uniform vec2 view;  // the canvas's size
out vec4 colour;

void main() {
  vec2 size = vec2(textureSize(slice, 0));
  ivec2 p = ivec2(vec2(gl_FragCoord.x, view.y - gl_FragCoord.y) * size / view);
  colour = vec4(texelFetch(slice, p, 0)) / 255.0;
}`;

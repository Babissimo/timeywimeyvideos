/** GLSL ES 3.00 sources for filling a volume, slicing it and crossfading it. */
import type { VolumeFormat } from "./volume";

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

// Full-range BT.601: luma, Cb and Cr from colour and back, all in levels from 0 to 255, Cb
// and Cr about 128.
const YCC = `
const float KR = 0.299, KB = 0.114, KG = 1.0 - KR - KB;
const float CB = 2.0 * (1.0 - KB), CR = 2.0 * (1.0 - KR);

vec3 ycc(vec3 rgb) {
  float y = dot(rgb, vec3(KR, KG, KB));
  return vec3(y, vec2(rgb.b - y, rgb.r - y) / vec2(CB, CR) + 128.0);
}

// ycc's inverse.
vec3 rgb(vec3 ycc) {
  float y = ycc.x, b = y + CB * (ycc.y - 128.0), r = y + CR * (ycc.z - 128.0);
  return vec3(r, (y - KR * r - KB * b) / KG, b);
}
`;

// A frame to convert, in an RGBA8 texture, row 0 its top.
const FRAME = `
uniform highp sampler2D frame;
out vec4 colour;

vec3 pixel(ivec2 p) {
  return round(texelFetch(frame, p, 0).rgb * 255.0);
}
`;

/** A frame's luma, into an R8 layer the frame's size. */
export const LUMA = `${HEADER}${YCC}${FRAME}
void main() {
  colour = vec4(floor(ycc(pixel(ivec2(gl_FragCoord.xy))).x + 0.5) / 255.0);
}`;

/**
 * A frame's chroma, into an RG8 layer half its size each way, rounded up: for each 2×2 block
 * of pixels, Cb and Cr of their mean colour. A block cut short by an odd size repeats the
 * pixels it has.
 */
export const CHROMA = `${HEADER}${YCC}${FRAME}
void main() {
  ivec2 p = 2 * ivec2(gl_FragCoord.xy), last = textureSize(frame, 0) - 1;
  vec3 sum = vec3(0.0);
  for (int k = 0; k < 4; k++) sum += pixel(min(p + ivec2(k & 1, k >> 1), last));
  colour = vec4(clamp(floor(ycc(sum / 4.0).yz + 0.5), 0.0, 255.0) / 255.0, 0.0, 1.0);
}`;

// How a slicing shader reads each format: voxel(t, y, x) gives a voxel's four levels, and
// shade(v) makes a pixel of them, or of a blend of them, v in units of 2^-12 of a level.
const READ: Record<VolumeFormat, string> = {
  // timeslice rounds a level v as min(255, int(v + 0.5)).
  rgba: `${VOXEL}
uvec4 shade(ivec4 v) {
  return uvec4(clamp((v + 2048) >> 12, 0, 255));
}
`,
  // Luma, Cb, Cr and 255, made colour only once blended.
  yuv420: `${YCC}
uniform highp sampler2DArray luma, chroma;  // chroma's texel (x, y) covers luma's (2x, 2y)

ivec4 voxel(int t, int y, int x) {
  vec2 c = texelFetch(chroma, ivec3(x >> 1, y >> 1, t), 0).rg;
  return ivec4(round(vec4(texelFetch(luma, ivec3(x, y, t), 0).r, c, 1.0) * 255.0));
}

uvec4 shade(ivec4 v) {
  return uvec4(uvec3(clamp(floor(rgb(vec3(v.xyz) / 4096.0) + 0.5), 0.0, 255.0)), 255u);
}
`,
};

const BLACK = "uvec4(0u, 0u, 0u, 255u)";

// timeslice blends voxels a = (t0, x0), b = (t0, x1), c = (t1, x0) and d = (t1, x1) in
// float64 as v = (a (1 - fx) + b fx) (1 - ft) + (c (1 - fx) + d fx) ft. Here v = a + fx (b - a)
// + ft (c - a) + fx ft (a - b - c + d) is summed exactly in integers, to v in units of 2^-12,
// with fx, ft and fx ft truncated to whole numbers of 2^-36, in three 12-bit parts, w[0] the
// most significant; the floored carries floor the whole sum. So a level rounded from it
// differs from timeslice's only where v is within about 1e-8 of halfway between two. In
// floats the compiler may reorder the sums and round differently; in integers the
// order makes no difference.
const BLEND = `
ivec4 blend(ivec4 a, ivec4 b, ivec4 c, ivec4 d, ivec3 w[3]) {
  ivec4 e1 = b - a, e2 = c - a, e3 = a - b - c + d;
  ivec4 sum = ivec4(0);
  for (int i = 2; i >= 0; i--) sum = (sum >> 12) + w[i].x * e1 + w[i].y * e2 + w[i].z * e3;
  return (a << 12) + sum;  // sum is v - a in units of 2^-12
}
`;

/** A slicing shader's source for a volume of each format, from its declarations and main. */
const slicing = (body: string): Record<VolumeFormat, string> => ({
  rgba: `${HEADER}${READ.rgba}${BLEND}${body}`,
  yuv420: `${HEADER}${READ.yuv420}${BLEND}${body}`,
});

/** A flat slice: output column j reads the voxels and weights worked out for it. */
export const FLAT = slicing(`
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
    colour = shade(voxel(at.x, row, at.z) << 12);
  } else {
    ivec3 w[3];
    for (int i = 0; i < 3; i++) w[i] = texelFetch(columns, ivec2(j, i + 1), 0).xyz;
    colour = shade(blend(voxel(at.x, row, at.z), voxel(at.x, row, at.w),
                         voxel(at.y, row, at.z), voxel(at.y, row, at.w), w));
  }
}`);

/**
 * A slice pushed off its plane by noise: each pixel works out its own noise and where that
 * moves it, then reads the volume there, as timeslice._sample_noisy_columns does.
 */
export const NOISY = slicing(`
uniform highp isampler2D columns;      // per column: t[j] and x[j] floored, first node column
uniform highp sampler2D offsets;       // per column: t[j] and x[j] past those; row 1 node weights
uniform highp isampler2D rows;         // per row: first node row
uniform highp sampler2D rowWeights;    // per row: node weights
uniform highp sampler2D nodes;         // the noise at the nodes
uniform vec2 push;                     // the (t, x) move where the noise is strongest
uniform ivec2 size;                    // the volume's frames and width
uniform bool nearest, wrap, wrapX;
out uvec4 colour;

float node(int column, int row) {
  return texelFetch(nodes, ivec2(column, row), 0).x;
}

// The noise at pixel (j, row), as timeslice._blend_row works it out: the 4 rows of nodes
// around it blended into one, then the 4 nodes of that around it, clamped to -1 to 1.
float bump(int j, int row, int first) {
  int r = texelFetch(rows, ivec2(row, 0), 0).x;
  vec4 down = texelFetch(rowWeights, ivec2(row, 0), 0);
  vec4 across = texelFetch(offsets, ivec2(j, 1), 0);
  float v = 0.0;
  for (int k = 0; k < 4; k++) {
    int m = first + k;
    v += across[k] * (down.x * node(m, r) + down.y * node(m, r + 1)
                      + down.z * node(m, r + 2) + down.w * node(m, r + 3));
  }
  return clamp(v, -1.0, 1.0);
}

// Whether lo + f lies in [-0.5, n - 0.5), for whole lo and f from 0 to 1.
bool within(int lo, float f, int n) {
  return (lo > -1 || (lo == -1 && f >= 0.5)) && (lo < n - 1 || (lo == n - 1 && f < 0.5));
}

// a mod n, from 0 to n - 1. GLSL leaves % undefined for negative operands.
int floorMod(int a, int n) {
  return a >= 0 ? a % n : n - 1 - (-1 - a) % n;
}

// As timeslice._neighbours: the voxels either side of lo + f along an axis of n.
ivec2 neighbours(int lo, int n, bool ring) {
  return ring ? ivec2(floorMod(lo, n), floorMod(lo + 1, n)) : ivec2(max(lo, 0), min(lo + 1, n - 1));
}

void main() {
  int j = int(gl_FragCoord.x), row = int(gl_FragCoord.y);
  ivec4 at = texelFetch(columns, ivec2(j, 0), 0);
  // Where the pixel reads, as a whole number and a fraction of a frame or column, relative
  // to its column's floors so that float32 keeps the fractions close to float64's.
  vec2 offset = texelFetch(offsets, ivec2(j, 0), 0).xy + push * bump(j, row, at.z);
  vec2 whole = floor(offset);
  float ft = offset.x - whole.x, fx = offset.y - whole.y;
  int t = at.x + int(whole.x), x = at.y + int(whole.y);
  if (!((wrap || within(t, ft, size.x)) && (wrapX || within(x, fx, size.y)))) {
    colour = ${BLACK};
    return;
  }
  ivec2 ts = neighbours(t, size.x, wrap), xs = neighbours(x, size.y, wrapX);
  if (nearest) {
    colour = shade(voxel(ft >= 0.5 ? ts.y : ts.x, row, fx >= 0.5 ? xs.y : xs.x) << 12);
    return;
  }
  vec3 rest = vec3(fx, ft, fx * ft);  // into blend's parts, fx ft already rounded to float32
  ivec3 w[3];
  for (int i = 0; i < 3; i++) {
    rest *= 4096.0;
    w[i] = ivec3(floor(rest));
    rest -= floor(rest);
  }
  colour = shade(blend(voxel(ts.x, row, xs.x), voxel(ts.x, row, xs.y),
                       voxel(ts.y, row, xs.x), voxel(ts.y, row, xs.y), w));
}`);

// A frame of a fade, drawn into an RGBA32UI scratch as whole levels: texel (j, y) holds bytes
// 16j to 16j + 15 of row y of a plane of `channels` bytes a texel, each uint's lowest byte
// first, and 0 past `width` texels. The shader gives faded(y, x), texel x of row y faded.
const PACK = `
uniform int width;     // the plane's texels faded in each row
uniform int channels;  // its bytes a texel: 4, 2 or 1
out uvec4 colour;

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int per = 4 / channels;  // texels to a uint
  uvec4 words = uvec4(0u);
  for (int k = 0; k < 4; k++) {
    for (int i = 0; i < per; i++) {
      int x = (4 * p.x + k) * per + i;
      if (x >= width) break;
      uvec4 level = faded(p.y, x);
      for (int c = 0; c < channels; c++) words[k] |= level[c] << (8 * (i * channels + c));
    }
  }
  colour = words;
}
`;

/**
 * One frame of a crossfade, as timeslice.crossfade works it out in float32: slice i of the
 * fade, at w of itself and 1 - w of slice i + past, rounded by adding 0.5 and truncating.
 */
export const CROSSFADE = `${HEADER}${VOXEL}
uniform highp sampler2D weights;  // for slice i: w and 1 - w
uniform int layer;                // the frame drawn
uniform ivec2 past;               // (t, x) from slice i to the slice it takes over from
uniform bool across;              // slice i is the frame's column i, not the frame
uniform int zero;                 // always 0, which the compiler can't know

// x as it is, though the compiler can't tell, so it rounds each product on its own as numpy
// does, where it might otherwise fuse one into a multiply-add.
vec4 rounded(vec4 x) {
  return intBitsToFloat(floatBitsToInt(x) ^ zero);
}

uvec4 faded(int y, int x) {
  vec2 w = texelFetch(weights, ivec2(across ? x : layer, 0), 0).xy;
  vec4 mixed = rounded(vec4(voxel(layer + past.x, y, x + past.y)) * w.y);
  mixed = rounded(mixed + rounded(vec4(voxel(layer, y, x)) * w.x));
  return min(uvec4(floor(mixed + 0.5)), 255u);
}
${PACK}`;

/**
 * One frame of a crossfade of columns in a chroma plane, each of whose columns covers two of
 * the picture's: the mean, over those of them the faded picture keeps, of the chroma each
 * takes, which for the first n is blended as CROSSFADE blends them, and past those is kept.
 */
export const CHROMA_CROSSFADE = `${HEADER}${VOXEL}
uniform highp sampler2D weights;  // for the picture's column i: w and 1 - w
uniform int layer;                // the frame drawn
uniform int n, kept;              // the picture's columns faded, and those it keeps

uvec4 faded(int y, int j) {
  vec4 own = vec4(voxel(layer, y, j)), sum = vec4(0.0);
  float count = 0.0;
  for (int x = 2 * j; x < min(2 * j + 2, kept); x++) {
    vec2 w = texelFetch(weights, ivec2(min(x, n - 1), 0), 0).xy;
    sum += x < n ? own * w.x + vec4(voxel(layer, y, (x + kept) >> 1)) * w.y : own;
    count += 1.0;
  }
  return min(uvec4(floor(sum / count + 0.5)), 255u);
}
${PACK}`;

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

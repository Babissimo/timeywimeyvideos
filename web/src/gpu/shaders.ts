/** GLSL ES 3.00 sources for filling a volume and slicing it. */
import { NOISE_BOUND } from "../noise";

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

// Ken Perlin's improved noise (2002), as noise.ts describes it, for the slice's bumps.
const NOISE = `
uniform highp isampler2D hash;  // the lattice hash, 0 to 255 shuffled and written out twice
uniform vec2 push;              // the (t, x) move where the noise is strongest: 0 for none
uniform float scale;            // lattice cells a pixel
uniform float depth;            // where the frame lies along the third axis, in cells
uniform int ring;               // the cells along that axis before the noise repeats

int perm(int i) {
  return texelFetch(hash, ivec2(i, 0), 0).r;
}

float ease(float t) {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// The dot product of d with one of the 12 vectors from a cube's centre to the middles of its
// edges, picked by the hash h.
float grad(int h, vec3 d) {
  h &= 15;
  float u = h < 8 ? d.x : d.y;
  float v = h < 4 ? d.y : h == 12 || h == 14 ? d.x : d.z;
  return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

// The noise at p, 0 <= p.z < ring.
float perlin(vec3 p) {
  vec3 cell = floor(p), d = p - cell;
  vec3 w = vec3(ease(d.x), ease(d.y), ease(d.z));
  int x = int(cell.x) & 255, y = int(cell.y) & 255;
  int z0 = int(cell.z) % ring & 255, z1 = (int(cell.z) + 1) % ring & 255;
  int a = perm(x) + y, b = perm(x + 1) + y;
  int aa = perm(a), ab = perm(a + 1), ba = perm(b), bb = perm(b + 1);
  return mix(mix(mix(grad(perm(aa + z0), d), grad(perm(ba + z0), d - vec3(1, 0, 0)), w.x),
                 mix(grad(perm(ab + z0), d - vec3(0, 1, 0)),
                     grad(perm(bb + z0), d - vec3(1, 1, 0)), w.x), w.y),
             mix(mix(grad(perm(aa + z1), d - vec3(0, 0, 1)),
                     grad(perm(ba + z1), d - vec3(1, 0, 1)), w.x),
                 mix(grad(perm(ab + z1), d - vec3(0, 1, 1)),
                     grad(perm(bb + z1), d - vec3(1, 1, 1)), w.x), w.y), w.z);
}

// The noise at pixel (j, y), from -1 to 1.
float bump(int j, int y) {
  return clamp(perlin(vec3(vec2(j, y) * scale, depth)) / ${NOISE_BOUND.toFixed(4)}, -1.0, 1.0);
}
`;

/**
 * An output frame of a sweep, sliced from a volume, row 0 its top. Pixel (j, y) reads the clip
 * at origin + j along, pushed off that line by the noise there, and blends the four voxels
 * around that point; outside the clip it is black, unless that axis wraps round. A loop fade
 * blends the clip's last n frames into its first and drops them, frame i keeping (i + 1) /
 * (n + 1) of itself; a side fade does the same with the picture's last columns.
 */
export const SLICE = `${HEADER}${YCC}${NOISE}
uniform highp sampler2DArray luma, chroma;  // chroma's texel (x, y) covers luma's (2x, 2y)
uniform vec2 origin, along;  // where column 0 reads, as (t, x), and the move to the next column
uniform ivec2 size;          // the clip's frames and columns, once faded
uniform ivec2 fade;          // the frames and columns faded
uniform bool wrap, wrapX;    // time, or x, runs round in a ring
uniform bool nearest;        // copy the voxel closest to the point rather than blend
out vec4 colour;

// Voxel (t, y, x) as luma, Cb and Cr, in levels from 0 to 255.
vec3 voxel(int t, int y, int x) {
  return 255.0 * vec3(texelFetch(luma, ivec3(x, y, t), 0).r,
                      texelFetch(chroma, ivec3(x >> 1, y >> 1, t), 0).rg);
}

// Voxel (t, y, x) with the picture's last columns faded into its first.
vec3 sideFaded(int t, int y, int x) {
  vec3 v = voxel(t, y, x);
  if (x < fade.y) v = mix(voxel(t, y, x + size.y), v, float(x + 1) / float(fade.y + 1));
  return v;
}

// The same with the clip's last frames faded into its first too.
vec3 faded(int t, int y, int x) {
  vec3 v = sideFaded(t, y, x);
  if (t < fade.x) v = mix(sideFaded(t + size.x, y, x), v, float(t + 1) / float(fade.x + 1));
  return v;
}

// a mod n, from 0 to n - 1. GLSL leaves % undefined for negative operands.
int floorMod(int a, int n) {
  return a >= 0 ? a % n : n - 1 - (-1 - a) % n;
}

// The voxels before and after p along an axis of n, wrapping round or held at the ends.
ivec2 around(float p, int n, bool wraps) {
  int lo = int(floor(p));
  return wraps ? ivec2(floorMod(lo, n), floorMod(lo + 1, n))
               : clamp(ivec2(lo, lo + 1), 0, n - 1);
}

// Whether p is in an axis of n voxels, or within half a voxel of its ends.
bool within(float p, int n) {
  return -0.5 <= p && p < float(n) - 0.5;
}

void main() {
  int j = int(gl_FragCoord.x), y = int(gl_FragCoord.y);
  vec2 p = origin + float(j) * along;  // (t, x)
  if (push != vec2(0.0)) p += push * bump(j, y);
  if (!((wrap || within(p.x, size.x)) && (wrapX || within(p.y, size.y)))) {
    colour = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }
  ivec2 t = around(p.x, size.x, wrap), x = around(p.y, size.y, wrapX);
  vec2 f = fract(p);
  vec3 v;
  if (nearest) {
    v = faded(f.x < 0.5 ? t.x : t.y, y, f.y < 0.5 ? x.x : x.y);
  } else {
    v = mix(mix(faded(t.x, y, x.x), faded(t.x, y, x.y), f.y),
            mix(faded(t.y, y, x.x), faded(t.y, y, x.y), f.y), f.x);
  }
  colour = vec4(floor(clamp(rgb(v), 0.0, 255.0) + 0.5) / 255.0, 1.0);
}`;

/** The slice scaled to the canvas, nearest pixel, its first row at the top. */
export const SHOW = `${HEADER}
uniform highp sampler2D slice;
uniform vec2 view;  // the canvas's size
out vec4 colour;

void main() {
  vec2 size = vec2(textureSize(slice, 0));
  ivec2 p = ivec2(vec2(gl_FragCoord.x, view.y - gl_FragCoord.y) * size / view);
  colour = texelFetch(slice, p, 0);
}`;

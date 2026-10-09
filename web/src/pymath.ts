// Python's arithmetic where JavaScript's differs, so the planner and the noise give exactly
// the numbers and messages timeslice.py does.

import type { Rate } from "./types";

export const TAU = 2 * Math.PI;

const scratch = new DataView(new ArrayBuffer(8));  // a double's bytes, read back as integers

/** A finite double's exact value as digits × 10^exp, with exp ≤ 0. */
function exactDecimal(x: number): { digits: bigint; exp: number } {
  scratch.setFloat64(0, Math.abs(x));
  const bits = scratch.getBigUint64(0);
  const biased = Number(bits >> 52n);
  let mantissa = bits & 0xfffffffffffffn;
  let twos = -1074;  // subnormal
  if (biased !== 0) {
    mantissa |= 1n << 52n;
    twos = biased - 1075;
  }
  if (twos >= 0) return { digits: mantissa << BigInt(twos), exp: 0 };
  // m / 2^k = m × 5^k / 10^k
  return { digits: mantissa * 5n ** BigInt(-twos), exp: twos };
}

const POWERS_OF_10 = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13,
                      1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22];  // all exact

/** n / d rounded to the nearest integer, halves to even; n ≥ 0, d > 0. */
function divideHalfEven(n: bigint, d: bigint): bigint {
  const q = n / d;
  const twice = 2n * (n % d);
  return twice > d || (twice === d && q % 2n === 1n) ? q + 1n : q;
}

function negative(x: number): boolean {
  return x < 0 || Object.is(x, -0);
}

/**
 * Python's round(x, digits): to the nearest multiple of 10^-digits, halves to even, judged on
 * the exact binary value (so 2.675 goes down to 2.67). With digits 0 it is round(x), an
 * integer, so never -0.
 */
export function roundHalfEven(x: number, digits = 0): number {
  if (!Number.isFinite(x)) return x;
  if (digits === 0) {
    let r = Math.round(x);
    if (Math.abs(x - r) === 0.5) r = 2 * Math.round(x / 2);
    return r === 0 ? 0 : r;
  }
  if (digits > 323) return x;
  if (digits < -308) return 0 * x;
  if (digits > 0 && digits < POWERS_OF_10.length) {
    // In floating point unless |x| × 10^digits might fall the other side of a half from the
    // rounded product. q and 10^digits are exact, so q / 10^digits is correctly rounded.
    const scale = POWERS_OF_10[digits];
    const y = Math.abs(x) * scale;
    const whole = Math.floor(y);
    if (y < 2 ** 52 && Math.abs(y - whole - 0.5) > y * 1e-15) {
      const q = y - whole < 0.5 ? whole : whole + 1;
      return negative(x) ? -(q / scale) : q / scale;
    }
  }
  const { digits: whole, exp } = exactDecimal(x);
  const shift = exp + digits;
  const q = shift >= 0
    ? whole * 10n ** BigInt(shift)
    : divideHalfEven(whole, 10n ** BigInt(-shift));
  return Number(`${negative(x) ? "-" : ""}${q}e${-digits}`);
}

/** Python's a % n: the result takes the sign of n (and a zero is +0, as for Python's ints). */
export function floorMod(a: number, n: number): number {
  const m = a % n;
  if (m === 0) return 0;
  return (m < 0) !== (n < 0) ? m + n : m;
}

/** Python's math.remainder: x less the nearest multiple of y, halves to the even multiple. */
export function ieeeRemainder(x: number, y: number): number {
  if (Number.isFinite(x) && Number.isFinite(y)) {
    if (y === 0) return NaN;
    const absx = Math.abs(x), absy = Math.abs(y);
    const m = absx % absy;
    // m against half of absy, compared as m against absy - m, which is exact where it matters.
    const c = absy - m;
    let r: number;
    if (m < c) r = m;
    else if (m > c) r = -c;
    else r = m - 2 * ((0.5 * (absx - m)) % absy);
    return copysign(1, x) * r;
  }
  if (Number.isNaN(x)) return x;
  if (Number.isNaN(y)) return y;
  if (!Number.isFinite(x)) return NaN;
  return x;
}

/** Python's math.gcd of two integers, never negative. */
export function gcd(a: number, b: number): number {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b) [a, b] = [b, a % b];
  return a;
}

function floorDiv(n: bigint, d: bigint): bigint {
  const q = n / d;
  return (n % d !== 0n) && ((n < 0n) !== (d < 0n)) ? q - 1n : q;
}

/**
 * Python's Fraction(x).limit_denominator(max): the closest fraction to x with a denominator of
 * at most max.
 */
export function limitDenominator(x: number, max = 1000000): Rate {
  if (max < 1) throw new RangeError("max_denominator should be at least 1");
  if (!Number.isFinite(x)) throw new RangeError(`cannot convert ${x} to a fraction`);
  // x exactly, as numerator / denominator in lowest terms (the denominator a power of 2).
  const { digits, exp } = exactDecimal(x);
  let num = negative(x) ? -digits : digits;
  let den = 10n ** BigInt(-exp);
  const divisor = bigGcd(num < 0n ? -num : num, den);
  num /= divisor;
  den /= divisor;
  const limit = BigInt(max);
  if (den <= limit) return { num: Number(num), den: Number(den) };

  let [p0, q0, p1, q1] = [0n, 1n, 1n, 0n];
  let [n, d] = [num, den];
  for (;;) {
    const a = floorDiv(n, d);
    const q2 = q0 + a * q1;
    if (q2 > limit) break;
    [p0, q0, p1, q1] = [p1, q1, p0 + a * p1, q2];
    [n, d] = [d, n - a * d];
  }
  const k = floorDiv(limit - q0, q1);
  // The nearer of (p0 + k p1) / (q0 + k q1) and p1 / q1.
  if (2n * d * (q0 + k * q1) <= den) return { num: Number(p1), den: Number(q1) };
  return { num: Number(p0 + k * p1), den: Number(q0 + k * q1) };
}

function bigGcd(a: bigint, b: bigint): bigint {
  while (b) [a, b] = [b, a % b];
  return a;
}

/** x × y as a sum hi + lo (Dekker), exact barring overflow and underflow. */
function exactProduct(x: number, y: number): [number, number] {
  const split = (v: number): [number, number] => {
    const t = v * 134217729;  // 2^27 + 1
    const hi = t - (t - v);
    return [hi, v - hi];
  };
  const [xh, xl] = split(x);
  const [yh, yl] = split(y);
  const p = xh * yh;
  const q = xh * yl + xl * yh;
  const z = p + q;
  return [z, p - z + q + xl * yl];
}

/**
 * Python's math.hypot: the length of a vector, almost always correctly rounded, so it can
 * differ from Math.hypot in the last bit.
 */
export function hypot(...coords: number[]): number {
  let max = 0;
  let nan = false;
  const vec = coords.map((c) => {
    const x = Math.abs(c);
    nan ||= Number.isNaN(x);
    if (x > max) max = x;
    return x;
  });
  return vectorNorm(vec, max, nan);
}

function vectorNorm(vec: number[], max: number, nan: boolean): number {
  if (max === Infinity) return max;
  if (nan) return NaN;
  if (max === 0 || vec.length <= 1) return max;
  const maxE = frexpExponent(max);
  if (maxE < -1023) {
    const DBL_MIN = 2.2250738585072014e-308;
    return DBL_MIN * vectorNorm(vec.map((x) => x / DBL_MIN), max / DBL_MIN, nan);
  }
  const scale = 2 ** -maxE;
  let csum = 1, frac1 = 0, frac2 = 0;
  const add = (hi: number) => {
    const sum = csum + hi;
    frac2 += (csum - sum) + hi;
    csum = sum;
  };
  for (const v of vec) {
    const x = v * scale;
    const [hi, lo] = exactProduct(x, x);
    add(hi);
    frac1 += lo;
  }
  let h = Math.sqrt(csum - 1 + (frac1 + frac2));
  const [hi, lo] = exactProduct(-h, h);
  add(hi);
  frac1 += lo;
  const x = csum - 1 + (frac1 + frac2);
  h += x / (2 * h);
  return h / scale;
}

/** The exponent e of C's frexp: x = m × 2^e with 0.5 ≤ m < 1, for finite x > 0. */
function frexpExponent(x: number): number {
  scratch.setFloat64(0, x);
  const biased = (scratch.getUint16(0) >> 4) & 0x7ff;
  if (biased === 0) return frexpExponent(x * 2 ** 64) - 64;
  return biased - 1022;
}

const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;

/** Python's math.radians, which multiplies by π/180 rather than dividing by 180. */
export function radians(x: number): number {
  return x * DEG_TO_RAD;
}

/** Python's math.degrees. */
export function degrees(x: number): number {
  return x * RAD_TO_DEG;
}

/** Python's math.copysign: x's size with y's sign, including the sign of a zero. */
export function copysign(x: number, y: number): number {
  return negative(y) ? -Math.abs(x) : Math.abs(x);
}

/**
 * Python's format(x, f".{precision}g"): precision significant digits, halves to even on the
 * exact value, trailing zeros dropped, and an exponent outside 1e-4 to 10^precision.
 */
export function formatG(x: number, precision = 6): string {
  if (Number.isNaN(x)) return "nan";
  const sign = negative(x) ? "-" : "";
  if (!Number.isFinite(x)) return `${sign}inf`;
  if (x === 0) return `${sign}0`;
  const p = Math.max(1, precision);
  const { digits, exp } = exactDecimal(x);
  let s = digits.toString();
  let e = s.length - 1 + exp;  // the power of 10 of the first digit
  if (s.length > p) {
    s = divideHalfEven(digits, 10n ** BigInt(s.length - p)).toString();
    if (s.length > p) {  // rounded up to the next power of 10
      s = s.slice(0, p);
      e += 1;
    }
  }
  s = s.replace(/0+$/, "");
  if (e < -4 || e >= p) {
    const mantissa = s.length > 1 ? `${s[0]}.${s.slice(1)}` : s;
    return `${sign}${mantissa}e${e < 0 ? "-" : "+"}${String(Math.abs(e)).padStart(2, "0")}`;
  }
  if (e < 0) return `${sign}0.${"0".repeat(-e - 1)}${s}`;
  const whole = s.slice(0, e + 1).padEnd(e + 1, "0");
  const fraction = s.slice(e + 1);
  return `${sign}${whole}${fraction ? `.${fraction}` : ""}`;
}

/**
 * Python's repr of a string: quoted, with backslashes, quotes, and the C0 control characters
 * and DEL escaped. Python also escapes other characters it can't print (U+00A0, say), which
 * this leaves as they are.
 */
export function pyRepr(s: string): string {
  const quote = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = quote;
  for (const ch of s) {
    const code = ch.codePointAt(0)!;
    if (ch === quote || ch === "\\") out += `\\${ch}`;
    else if (ch === "\t") out += "\\t";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (code < 0x20 || code === 0x7f) out += `\\x${code.toString(16).padStart(2, "0")}`;
    else out += ch;
  }
  return out + quote;
}

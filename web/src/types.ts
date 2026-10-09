/** A frame rate as a fraction, like Python's Fraction (e.g. 30000/1001). */
export interface Rate { num: number; den: number }

/** Where each column of one output frame reads the volume: time t[j] and position x[j]. */
export interface Columns { t: Float64Array; x: Float64Array }

/**
 * The noise over one output frame, as timeslice.Noise._grid gives it: values at the
 * nodes (already divided by NOISE_BOUND), row-major nodeRows × nodeCols, and for each
 * pixel column and row the first of the 4 nodes around it and their 4 weights.
 */
export interface NoiseGrid {
  nodes: Float64Array; nodeRows: number; nodeCols: number;
  cols: Int32Array; colW: Float64Array;   // width, width × 4
  rows: Int32Array; rowW: Float64Array;   // height, height × 4
}

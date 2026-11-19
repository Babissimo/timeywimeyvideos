// Whether the page can work here, and the state of the video chosen, which every part reads.

import type { ClipInfo } from "../decode";
import type { FrameInfo } from "../live";

// Why this page can't decode or encode video here, if it can't. WebCodecs, which does both,
// runs only in a secure context.
export const UNABLE = !window.isSecureContext ?
  `The page can only decode video in a secure context, which ${location.origin} isn't: ` +
  "open it at localhost or over HTTPS." :
  !("VideoDecoder" in window && "VideoEncoder" in window) ?
  "This browser can't decode or encode video in a page, as it has no WebCodecs. Use a " +
  "recent Chrome or Edge." : "";

export const state: {
  source: File | null; id: string; chosen: number; clip: ClipInfo | null;
  info: FrameInfo | null; pos: number; playing: boolean; scrubbing: boolean;
} = {
  source: null,    // the File chosen
  id: "",          // and its id in the list
  chosen: 0,       // how many times a source has been chosen
  clip: null,      // what the chosen file holds, once probed
  info: null,      // the live view's description of the frame on screen
  pos: 0.5,        // how far through the output video, 0 to 1
  playing: false,
  scrubbing: false,
};

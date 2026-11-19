// The elements of index.html the page looks up by id, and the tag each has there. The type
// check holds the page to these ids, and elements.test.ts holds them to index.html.
export const TAGS = {
  angle: "input", "angle-range": "input", duration: "input", file: "input", folder: "input",
  fps: "input", "loop-fade": "input", "loop-sides": "input", noise: "input",
  "noise-amount": "input", "noise-range": "input", "noise-seed": "input",
  "noise-size": "input", "noise-speed": "input", scale: "input", scrub: "input",
  "side-fade": "input", start: "input",
  motion: "select", "noise-direction": "select", source: "select",
  cancel: "button", "live-action": "button", "open-folder": "button", play: "button",
  "render-full": "button", "render-preview": "button",
  cube: "canvas", live: "canvas",
  player: "video",
  renders: "ul",
  drop: "label",
  "frame-label": "span", "live-info": "span", "now-playing": "span",
  job: "div", "job-bar": "div", "live-message": "div", "live-stage": "div",
  "loop-fields": "div", "motion-field": "div", "noise-fields": "div", "side-fade-field": "div",
  "cube-loop": "p", "cube-noise": "p", "edges-note": "p", estimate: "p", "folder-note": "p",
  "job-text": "p", "live-text": "p", "motion-note": "p", "noise-note": "p", "slice-note": "p",
  "source-info": "p", unserved: "p",
} as const satisfies Record<string, keyof HTMLElementTagNameMap>;

type Tags = typeof TAGS;

/** The element of index.html with this id, typed by its tag. */
export function $<Id extends keyof Tags>(id: Id): HTMLElementTagNameMap[Tags[Id]] {
  return document.getElementById(id) as HTMLElementTagNameMap[Tags[Id]];
}

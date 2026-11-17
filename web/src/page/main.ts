// The page: open a video, set the slice, watch it play live, see where the
// slice sits in the cuboid, and render it. The video stays on this computer:
// the live view and the renders decode and slice it here, on the GPU.

// controls, view, cuboid, source and renders import one another, so each calls into the others
// only from start() and from listeners, never as it loads, when another may not have loaded yet.
import { noiseChanged, slicesChanged } from "./controls";
import { $ } from "./elements";
import { enableRender, restore } from "./renders";
import { NO_SOURCE, showSources } from "./source";
import { UNABLE } from "./state";
import { showMessage } from "./view";

function start() {
  $("unserved").remove();  // the note for when this script can't run
  slicesChanged();
  noiseChanged();
  restore();
  showSources();
  enableRender();  // as Firefox may keep a button's state from before a reload
  showMessage(UNABLE || NO_SOURCE);
}

start();

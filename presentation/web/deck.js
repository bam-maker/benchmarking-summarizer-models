const stage = document.querySelector(".reveal");
const pauseButton = document.querySelector("#pause-motion");
const deck = new Reveal(stage, {
  width:1280, height:720, margin:0.035, center:false, minScale:0.1,
  maxScale:2.5, controls:false, progress:true, hash:true, history:false,
  transition:"none", backgroundTransition:"none", navigationMode:"linear",
  keyboard:{80:toggleMotion, 82:replay}, help:false, autoSlide:0
});
function toggleMotion() {
  const paused = document.body.classList.toggle("halt");
  pauseButton.textContent = paused ? "Resume motion" : "Pause motion";
  pauseButton.setAttribute("aria-pressed", String(paused));
}
function replay() {
  const {h,v} = deck.getIndices();
  deck.slide(h,v,-1);
  const slide = deck.getCurrentSlide();
  for (const animation of slide.getAnimations({subtree:true})) {
    animation.currentTime = 0;
  }
}
function preparePrint() {
  const root = document.querySelector("#print-deck");
  root.replaceChildren(...Array.from(stage.querySelectorAll("section > .slide"), slide => slide.cloneNode(true)));
  root.querySelectorAll(".fragment").forEach(el => {
    el.classList.remove("fragment","current-fragment");
    el.removeAttribute("aria-hidden");
  });
}
stage.addEventListener("click", event => {
  if (event.defaultPrevented || deck.isOverview() || event.target.closest("a,button") || getSelection().toString()) return;
  deck.next();
});
pauseButton.addEventListener("click", toggleMotion);
document.querySelector("#replay").addEventListener("click", replay);
document.querySelector("#overview").addEventListener("click", () => deck.toggleOverview());
document.querySelector("#fullscreen").addEventListener("click", () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen();
});
window.addEventListener("beforeprint", preparePrint);
deck.initialize().then(() => {
  document.body.dataset.ready = "true";
});

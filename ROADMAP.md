# Kickstep roadmap

## Maybe: learning while it plays

A second Kickstep that waits ~200 ms (hears each clap's whole body) checks the live one's hits a moment later, and a
tiny part of the live model (last layer or per-drum sensitivity) adapts to the current song, fading back to the
shipped weights. Never self-graded (it would reinforce its own mistakes), never permanent (drift, unreproducible
bugs). First test: train a 200 ms-delay model offline; only worth building if it's clearly better on claps. That
model would also make a better offline teacher.

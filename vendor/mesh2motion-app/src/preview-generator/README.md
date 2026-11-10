# Preview Generator

The goal of this preview generator is to create small animated clip previews to use in the animation listing. It is fully done in the web browser. So that is the only "tool" needed.


## Workflow

1. Go to the URL manually for the preview generator once the web server is started (command: npm run dev): http://localhost:5173/preview-generator/index.html
2. Select a 3d model that you want to generate animations from the drop-down.
3. Rotate and position the 3d model
4. Press the record button
5. Wait for all the animations to finish playing/recording. A ZIP file will be downloaded when they are all done
6. Place the animation preview webm files in the static/animpreviews folder. Each skeleton type has its own folder. 

## Tech talk
In step 1 of the workflow, the model/animation selection pulls the GLB files from the static/animations folder. These files have all the animations as well as a skinned mesh, so we can focus on recording and not skinning.

The recording is all done with the "MediaRecorder API" that is a vanilla javascript API. The JSZIP javascript package does the zipping behavior.

The canvas size on the preview-generator is hard-coded in the preview-generator.ts file. The display on the preview generator is larger than the result will be, but it is easier to position when the canvas size is larger. The UI in the application expects a certain size the previews will be in. This helps with the lazy loading and the preview sizes are hard-coded in the Mesh2Motion app itself in CSS.
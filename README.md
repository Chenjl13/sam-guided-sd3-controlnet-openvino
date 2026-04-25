# sam-guided-sd3-controlnet-openvino
Summer internship

<img src="imgs/Sunglass_Cat.png">

## 3-phase-image-generating and inpainting
[Process](Process/readme.md) and [Code](Process/Code) are available here.

### Text to Image
Giving the text input, stable diffusion 3 can generate the matched image. 

### Mask Generation
Giving the image, SAM2 creates the mask based of chosen point or bbox.

### Inpainting
Giving the original image, its mask and prompt for replacing, stable diffusion 3 can inpaint the image.





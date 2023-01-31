# ImageAnnotation

## Description

At that time, we developed an image annotation system based on special medical tasks. The system supports image annotation and echo annotation information, as well as back-end data management, user management, role management, etc. Now we will open source the front-end offline version based on vue.

Image annotation mainly uses rectangles and polygons to annotate images and customize annotation information. It is divided into three main bodies. The left side mainly displays image information and label information, the middle is the specific operation area, and the right side is divided into control area, operation instructions and precautions, as shown in the following figure:

![](https://dorc.baai.ac.cn/imgs/projects/dorc-label-offline/11.png)

1. ## Installation

   1. To install nginx locally, [Go to](http://nginx.org/).

## Usage

### Quick start

1. Put the `flagdata/annotation/dist` folder under the default `html` of nginx.

2. Modify `nginx.confg` and add location.

   ```
   location / {
       root /{your html path}/dist;   # change
       index index.html index.htm;
       try_files $uri $uri/ /index.html;
   }
   ```

3. Restart nginx.

4. Access the IP address configured by nginx.

### Advanced Usage

1. The user clicks the "Upload" button on the right to upload the local **image folder**. After the upload is successful, the image list will be displayed on the left, as shown in the following figure:

   ![](https://dorc.baai.ac.cn/imgs/projects/dorc-label-offline/12.png)

2. The user can switch images in the image list on the left, and the selected image will be displayed in the middle annotation area. The user can zoom the image by Alt+mouse wheel, or move the image by Alt+left mouse button.

3. You can select relevant dimension graphics (rectangle or polygon), line color and fill color in the right control area for drawing, as shown in the following figure:

   ![](https://dorc.baai.ac.cn/imgs/projects/dorc-label-offline/13.png)

4. Press and hold the left mouse button to select the label box, and use the Delete key to delete the label box.

2. When marking polygons, use Ctrl+the left mouse button to display the points of the polygon. Click the points with the left mouse button to drag.

3. In the left area, the label list, click the "Operation" icon, and the user can modify the label information.

7. When you want to save the annotation information, just click the "Save" button in the operation area on the right, give the result as follows:

   ```json
   {
       "angle": 0,
       "height": 121.11,
       "width": 183.33,
       "left": 253.28,
       "top": 440.72,
       "scaleX": 1,
       "scaleY": 1,
       "strokeWidth": 1,
       "stroke": "#0439a9",
       "fill": "rgba(255, 255, 255, 0)",
       "descibe": "TagName1",
       "type": "polygon",
       "id": 1672369472552,
       "hidden": false,
       "opacity": 1,
       "path": [{"x": 304.8888888888889,"y": 444.55555555555554},{"x": 262.6666666666667,"y": 485.6666666666667},{"x": 263.77777777777777,"y": 535.6666666666666},{"x": 253.77777777777777,"y": 562.3333333333334},{"x": 341.55555555555554,"y": 550.1111111111111},{"x": 437.1111111111111,"y": 522.3333333333334},{"x": 398.22222222222223,"y": 460.1111111111111},{"x": 349.33333333333337,"y": 441.22222222222223}]
   }
   ```
   
8. When users want to echo the label information, they only need to click the "Echo" button in the right operation area to upload a single result file or a folder containing multiple results;

9. For specific operation and precautions, refer to the operation instructions and precautions on the right side.

## Extra Notes

The screenshots are all online pictures. If there is any infringement, please contact us and we will delete it immediately.
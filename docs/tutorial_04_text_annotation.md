# Text Annotation

## Description

The text annotation tool currently supports the annotation of **comparative relational text**. When uploading a local `txt` document, the browser will automatically load the text to be marked by line. Press and hold the left mouse button to select the text spans and its corresponding subjects, objects and attributes, as illustrated in the following figure:

![](https://dorc.baai.ac.cn/imgs/projects/dorc-label-offline/14.gif)

## Installation

To install nginx locally, please refer to the [official website](http://nginx.org/).

## Usage

### Quick start

1. Put the `flagdata/annotation/dist` folder under the default `html` of nginx.

2. Modify `nginx.confg` to add location.

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

1. The user clicks the "Upload Data" button to upload the local txt document, and the browser will automatically load the text to be marked by line.

2. Press and hold the left mouse button to select text, place the mouse on the selected text, click the right mouse button to select subject, object and attribute, as shown in the following figure:

   ![](https://dorc.baai.ac.cn/imgs/projects/dorc-label-offline/15.png)

3. After marking a comparison relationship, click the "Submit" button to start the next marking.

2. If you need to modify a marked item, click the "Modify" button to modify it.

3. The "Delete All" button will delete the information marked in the current corpus.

6. The user clicks the "Export Marked Data" button, and a json file containing the marked information will be generated locally, the contents are as follows:

   ```json
   [
       {
           "corpus": "I am fingding my way around this laptop better than my last one.",
           "index": 0,
           "markInfo": [
               {
                   "t1": "this laptop",
                   "t1_start": 28,
                   "t1_end": 39,
                   "t2": "last one",
                   "t2_start": 55,
                   "t2_end": 63,
                   "aspect_term": "",
                   "aspect_term_start": "",
                   "aspect_term_end": "",
                   "aspect": "GENE",
                   "status": 0,
                   "index": 0,
                   "option": {
                       "key": "BETTER",
                       "label": " BETTER"
                   }
               }
           ]
       }
   ]
   ```

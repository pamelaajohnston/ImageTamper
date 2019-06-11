# Video Tampering Problems and Detection

**TL;DR** This github was created to determine whether compression features can be used as an ally rather than an enemy to detect video tampering. Scroll down to the bottom for some notes on the code.

Video manipulation is a potential problem in today's digital society.
There are numerous convincing video manipulation techniques being developed or already in existence and we have no easy and reliable way of detecting tampering. There are tampering detection techniques, but these tend to be tied in to specific tampering methods. When we are confronted with tampered video content, we will not be able to see the tampering, let alone identify which tampering technique was used. For such a problem, it is important to develop techniques which are *tampering-type-agnostic*, that is, ways in which we can make the presence of manipulation visible.

## Video Compression
Video tampering and video compression have one important thing in common: they are both designed to be invisible to human eyes. Compression has always been designed to be invisible. Lossy compression in particular exploits weaknesses in human vision in order to reduce the number of bits required to represent the information. Take a look at these examples:

Uncompressed:

![GIF of Uncompression](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/mobile0_cif.gif)

Quantisation Parameter set to 14:

![GIF of QP=14](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/mobile14_cif.gif)

Quantisation Parameter set to 28:

![GIF of QP=28](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/mobile28_cif.gif)

Quantisation Parameter set to 42:

![GIF of QP=42](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/mobile42_cif.gif)

The first image is a GIF representation of one of the sequences from [Derf's media collection](https://media.xiph.org/video/derf/). The seqence is "mobile", the CIF version. In [Derf's] (https://media.xiph.org/video/derf/), it is supplied as YUV, but I've translated it to a GIF using my own `toGIF.py` script (which uses ImageMagick to do the GIF part, and I started with [this tutorial](http://superfluoussextant.com/making-gifs-with-python.html)). The other three are all compressed using H.264/AVC standard. They've been compressed using the constant bitrate mode of `x264` and levels of QP=[14, 28, 42]. You can see that QP=14 doesn't look particularly different from uncompressed. At QP=28, you start to lose some visible details: some of the vertical lines in the calendar, edges are just a *little* bit more blocky. And at QP=42, it's starting to actually look terrible.

### Detecting Compression Levels
Now, human eyes might not always be able to differentiate between levels of compression but *computer* eyes are somewhat different. I (and my co-authors Eyad Elyan and Chrisina Jayne) showed one way of doing this in the paper ["Toward Video Tampering Exposure: Inferring Compression Parameters from Pixels"](https://link.springer.com/chapter/10.1007/978-3-319-98204-5_4), published at EANN. Basically, you take a selection of video sequences, split them into "test" and "train" sets (to ensure no data leakage), encode (and decode) them using defined compression parameters, snip out some pixel patches and use the compression parameter as a label, then use that dataset to train a convolutional neural network. Obviously you should select hyper-parameters and network architecture appropriately, but you can look at my tensorflow code (all the files starting with `itNet...') to see some of the things that I tried. 

Here I only discuss quantisation parameter, but in the follow-up paper "[Video tampering localisation using features learned from authentic content](https://link.springer.com/article/10.1007/s00521-019-04272-z)" published in Neural Computing and Applications, we used another three features. Alone they weren't as effective as QP, but they did improve detection slightly when used together. In the examples of Quantisation Maps here, I've used an 80x80 pixel patch.

## Types of Video Tampering

There are several methods of video tampering, and in our paper "[A review of digital video tampering: From simple editing to full synthesis](https://www.sciencedirect.com/science/article/pii/S1742287618304146)" published in Digital Investigation, we arranged them as a spectrum of video tampering, from techniques which are somewhat limited in their content-changing abilities to fully synthetic video:

![Spectrum of Manipulation](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/fig_spectrum.png)

Examining a lot of different types of tampering techniques, particularly the state-of-the-art ones, we can see that what used to only be in the domain of special effects and cinematography wizards is slowly making its way down just about anyone with a computer.

### Inter-frame tampering
This is essentially invisible editing, and this type of tampering is already in the public domain. Examples range from [this](https://www.independent.co.uk/news/world/middle-east/gaza-protests-latest-idf-condemned-edited-video-angel-of-mercy-medic-razan-al-najjar-a8389611.html) story in The Independent, to faked Obama speeches reported in [Politifact](https://www.politifact.com/truth-o-meter/statements/2014/jun/23/chain-email/video-barack-obama-speech-circulating-internet-was/). When the source material is available, it is already possible to change the meaning or message of a video, and fashion whatever story you like out of it. These are examples that we know about. 

Here's an example I made myself:

Untampered original:

![Original Sequence](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/hall.gif)

Keep repeating the first 8 frames:

![Tampered Sequence](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/hall2.gif)

I've used ImageMagick again to make the GIFs and the original sequence is "hall" from [Derf's](https://media.xiph.org/video/derf/). If you look closely, you can see that I haven't quite erased the man completely, you can still see his foot. Repeating the first 8 frames of the sequence over and over again gives just enough motion that the CCTV-like video looks like it is live, but the man hasn't appeared. There are, however, other ways to remove things from videos.

### Object tampering

"Object Tampering" includes in-painting, spatio-temporal copy-move and splicing. It's called "object tampering" because it usually involves the addition or removal of a specific "object" in the video. 

Here's a snippet from a sequence from [Video Tampering Dataset](https://www.youtube.com/channel/UCZuuu-iyZvPptbIUHT9tMrA) (and its associated paper [Development of a video tampering dataset for forensic investigation by Al-Sanjary et al](https://doi.org/10.1016/j.forsciint.2016.07.013)).
Original:
![Original Sequence](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/manstreet_r.gif)

Tampered:
![Tampered Sequence](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/manstreet_f.gif)

This shows a spatio-temporal copy-move. Basically, *regions* of past or future frames are used to conceal the man but leave the moving car and person in the background in place.

This category includes chroma-keying, where an object is filmed in front of a green screen so that its background can be accurately removed. Then the object can be spliced on to a new background. D'Avino et al used chroma-keying to create their tampered video dataset in their paper "[Autoencoder with recurrent neural networks for video forgery detection](https://doi.org/10.2352/ISSN.2470-1173.2017.7.MWSF-330)", and here's a snipped from one of their examples:

![Tampered Sequence](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/hen_f.gif)

Object tampering techniques allow for greater content modification than inter-frame tampering techniques. You can remove things from the video completely or add things that were never there. Of course, these techniques have long been around in cinematography, but they were previously incredibly time-consuming and only really available via professional equipment.


### Synthetic video and Digital Puppetry

This is by far the most alarming group of techniques. And even more alarming is the fact that human beings are really not great at identifying some manipulated content. The study in [Faceforensics: A large-scale video dataset for forgery detection in human faces](https://arxiv.org/abs/1803.09179) revealed that humans did little better than guessing when trying to distinguish between forged and authentic content. It really isn't surprising when you see some examples:

![Tampered](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/z_ff_blueDude_640x480_1.gif)
![Authentic](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/z_ff_blueDude_640x480_1_ori.gif)
![Tampered](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/z_ff_woman_640x480_1.gif)
![Authentic](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/z_ff_woman_640x480_1_ori.gif)

That combined with [DeepFakes](https://www.bbc.co.uk/news/av/technology-43118477/deepfakes-the-face-swapping-software-explained) shows how digital puppetry, at least applied to the human face, is advancing very quickly. And [motion retargeting](https://www.youtube.com/watch?v=PCBTZh41Ris) has shown how similar digital puppetry can be performed on the whole body.


## Detecting Video Tampering using Compression Features

So, if the video tampering problem is potentially that big, and yet invisible, how do we tackle it? Well, one way to tackle it is to express the video content in a different way, specifically, as compression features. Earlier, I mentioned how I trained a convolutional neural network to detect compression parameters. These compression parameters, in particular the quantisation level turn out to be very good for detecting image manipulation:

![Tampered Tree](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/fig_tree_fake.png)
![Tampered Tree QP](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/fig_tree_qp.png)
This is a frame from one of the spliced sequence's from [D'Avino et al's dataset](https://doi.org/10.2352/ISSN.2470-1173.2017.7.MWSF-330). An extra tree has been added and it is not particularly noticable. A map of the predicted quantisation parameter, however, shows a particularly low level of compression level in the tampered area (the black region). 

Similarly, the hen, also from [D'Avino et al's dataset](https://doi.org/10.2352/ISSN.2470-1173.2017.7.MWSF-330), displays a lower level of compression around the tampered region:
![Tampered Hen](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/fig_hen_fake.png)
![Tampered Hen QP](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/fig_hen_qp.png)

And again for the FaceForensics dataset examples:

![Tampered Face](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/fig_ff_munge14_alt_pix.png)
![Tampered Face QP](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/fig_ff_munge14_alt_qp.png)
![Authentic Face](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/fig_ff_munge14_ori_pix.png)
![Authentic Face QP](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/ig_ff_munge14_ori_qp.png)

![Tampered Face](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/fig_ff_munge62_alt_pix.png)
![Tampered Face QP](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/fig_ff_munge62_alt_qp.png)
![Authentic Face](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/fig_ff_munge62_ori_pix.png)
![Authentic Face QP](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/fig_ff_munge62_ori_qp.png)

A quick scan through all the 640x480 test sequences from FaceForensics reveals that many tampered sequences display the same pattern: very low levels of compression around the manipulated area, but relatively high levels in other places (the tampered sequences and their QP feature maps are on the top).
![FaceForensics](https://github.com/pamelaajohnston/ImageTamper/blob/master/imagesAndGifs/z_mygif.gif)

In this way, region manipulation can be detected and localised across different tampered video datasets which use different tampering methods. Most importantly, the CNN used to create this representation was trained **only on authentic content**.




# Notes on the code in this repository
For the most part, I have not implemented command line arguments. Many files contain an "arguments" variable at a global level which I then repeatedly overwrite allowing me to keep track of what I've implemented. In such cases, you simply need to over-write that variable, too. I'll get round to tidying up the code at some point. Otherwise, here is a brief summary of what each file is for.

`YUVFrameToPNG.py` Takes a YUV 4:2:0 file (specified by filename, width, height, frame number and output name), extracts a frame and outputs it as .png (losslessly compressed - nice!).

`analyseClassifiers.ipynb` A notebook for analysing the results of some classifiers as pandas dataframes.

`analyseRegressionResults.ipynb` A notebook for analysing the results of some classifiers as pandas dataframes.

`averages.xlsx` analysis of VTD and D'Avino's datasets

`bin_to_blackAndWhite.py` a method to change a binary file into a black and white YUV file (make it visible)

`collateDuplicates.py` an analysis of a patched dataset to see how many duplicates were present

`collateYUVs.py` just appending a bunch of YUVs together (with some folder/file naming conventions)

`createDataset.py` Taking a load of YUV files and turning them into a patched dataset

`datasetViewing_intra.ipynb` Actually look at dataset patches

`dctCreation.ipynb` converting DCT coefficients to a spatial representation to look at (adapted from JM H.264 code DCT)

`detectDuplicates.py` an analysis of a patched dataset to see how many duplicates were present

`detectDuplicatePatches.py` an analysis of a patched dataset to see how many duplicates were present

`diffs.py` just do a diff

`encodeYUV.py` Take a bunch of YUV files and encode them using x264

`encodeYUVandEncodeAgain.py` Take a bunch of YUV files and encode them using x264, then recompress them

`faceforensics_createDataset.py` Turn the FaceForensic .avi files into YUV (and crop out the masked part for a balanced 
dataset)

`functions.py` mostly conversion between different YUV colour space and RGB, just a function library

`generateHeatmapFromYUV.py` Take a YUV file and (optionally) extract the compression features using a trained CNN, then 
turn the compression features into various graphs and combine them in order to detect tampering

`generateHeatmapFromYUV2.py` Take a YUV file and (optionally) extract the compression features using a trained CNN, then turn the compression features into various graphs and combine them in order to detect tampering, adapted slighlty for FaceForensics

`generateMasks....py` in a dataset where only real and tampered counterparts are provided, make a mask by doing a binary diff of the two.

`iNet....py` This is the CNN. It's for multiple GPUs but I only ever ran it on one. Requires TensorFlow 1.4 (IIRC)

`matFileRead.py` because some "video files" are actually matlab files

`normaliseAFile.py`

`patchIt.py` take a YUV file, create patches (temporal and spatial sampling)

`patchIt2.py` take a YUV file, create patches (slightly different temporal and spatial sampling)

`prettyPicture.py` just some nice smooth colours to make everything look decent.

`regression...` take some compression parameters, perform regression (i.e. tampering-type-specific classification)

`sobel.py` take an image, run a sobel kernel over it

`tamperYUV.py` combine two YUV sequences in a splice (not terrible results given how simple it is)

`test_train_split.py` split a dataset into test and train, assuming no duplicates in the dataset

`toGIF.py` transform a YUV or an AVI to a GIF

`toYUV.py`

`toYUV_PNG.py`

`toy.py` and other toys - generally just graph plotting or code snippets I want to keep. Useful to me and probably no one else.

`watershed.py` openCV's watershed segmentation.

`yuvview.py` another function library specifically dealing with viewing YUV files.

`imagesAndGifs` Folder containing some sample images and GIFs, usually from publicly available datasets (contact me if you 
need any of these removed)

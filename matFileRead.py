import scipy.io


if __name__ == "__main__":
    filenames = ["/Users/pam/Documents/data/TestVideos/VideoSeq_1.mat",
                 "/Users/pam/Documents/data/TestVideos/VideoSeq_2.mat",
                 "/Users/pam/Documents/data/TestVideos/VideoSeq_3.mat",
                 "/Users/pam/Documents/data/TestVideos/VideoSeq_4.mat"]

    for filename in filenames:
        mat = scipy.io.loadmat(filename)
        frames = mat['frames']
        print('Here is the frame for {}'.format(filename))
        print(frames.shape)
        # it's in format height, width, channels, frames
        height = frames.shape[0]
        width = frames.shape[1]
        channels = frames.shape[2]
        num_frames = frames.shape[3]


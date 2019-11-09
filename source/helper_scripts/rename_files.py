import os
import re
#                          ##################################################
#                          #########   TO BE RUN ON GOOGLE COLAB   ##########
#                          ##################################################


def main():
    id = 0

    folder_names = os.listdir('../data/')

    for f in folder_names:
        frame_names = os.listdir('../data/' + f)
        ordered_files = sorted(frame_names, key=lambda x: (int(re.sub('\D', '', x)), x))

        for fr in ordered_files:
            os.rename('../data/' + f + '/' + fr,
                      '../data/' + f + '/' + 'frame%d.png' % id
                      )
            id = id + 1


if __name__ == '__main__':
    main()

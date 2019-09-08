import os
from tkinter import *
from tkinter import messagebox, simpledialog

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from matplotlib import pyplot as plt


class MainWindow:

    def __init__(self, main, img_width, img_height, raw_frames, photo_images, frames_name, output_name):

        self.root = main

        # lists of points
        self.points = []

        # window's frame
        self.main_frame = Frame(main, height=img_height, width=img_width + 90, bd=15, relief=FLAT)

        # buttons' space
        self.cmd_frame = Frame(self.main_frame, height=img_height, width=90, bd=3, relief=SUNKEN)

        # canvas for image
        self.canvas = Canvas(self.main_frame, width=img_width, height=img_height, highlightthickness=0, bd=0)
        self.canvas.bind("<Button-1>", self.get_coord)  # left button click

        # images
        self.my_images = photo_images

        self.my_frames = raw_frames
        self.current_frame = 0

        # frames' file names
        self.filenames = frames_name

        # set first image on canvas
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=NW, image=self.my_images[self.current_frame])

        # name of the current frame
        self.curr_frame_name = Label(self.cmd_frame, text=self.filenames[self.current_frame])

        # frame count
        self.count = Label(self.cmd_frame, text='%d/%d' % (self.current_frame, len(self.filenames)))

        # button to change image
        self.button = Button(self.cmd_frame, text='Next >>', command=self.change_frame)

        # button to invalid frame
        self.invalid_btn = Button(self.cmd_frame, text='Invalid X', command=self.invalid_frame)

        # button to clear marked points
        self.clear_btn = Button(self.cmd_frame, text='Clear', command=self.clear_points)

        self.main_frame.pack(fill=BOTH)
        self.canvas.pack(side=LEFT)
        self.cmd_frame.pack(side=RIGHT, fill=BOTH)
        self.button.pack(expand=True)
        self.invalid_btn.pack(side='bottom')
        self.clear_btn.pack(side='bottom')
        self.curr_frame_name.pack(expand=True)
        self.count.pack(expand=True)

        self.output_file = '../../data/targets/' + output_name + '-v2.csv'

    def change_frame(self):
        if len(self.points) == 1:
            # save current coordinates
            f = open(self.output_file, 'a')
            txt = '%s;%s\n' % (
                self.filenames[self.current_frame], self.points[0])
            f.write(txt)
            f.close()

            # next image
            self.current_frame += 1

            # return to first image
            if self.current_frame == len(self.my_images):
                messagebox.showinfo('This is the End', 'The current sequence of frames is ended.')
                self.root.destroy()
                return

            # change image
            self.canvas.itemconfig(self.image_on_canvas, image=self.my_images[self.current_frame])
            self.curr_frame_name.configure(text=self.filenames[self.current_frame])
            self.count.configure(text = '%d/%d' % (self.current_frame, len(self.filenames)))
            self.points.clear()

    def get_coord(self, event):
        if len(self.points) < 1:
            x = event.x // 2
            y = event.y // 2
            # outputting x and y coord to console
            print(x, y, event.num)
            if event.num == 1:
                self.points.append((x, y))

        if len(self.points) == 1:
            # ########################## showing points for testing ###############################
            blank = np.copy(self.my_frames[self.current_frame])
            res = cv2.circle(blank, (self.points[0][0] * 2, self.points[0][1] * 2), 3, (255, 0, 0), -1, lineType=cv2.LINE_AA)

            plt.figure()
            plt.imshow(res)
            plt.show()
            # #####################################################################################
            print(self.points)

    def invalid_frame(self):
        # save current coordinates
        f = open(self.output_file, 'a')
        txt = '%s;%s\n' % (self.filenames[self.current_frame], '(-1, -1)')
        f.write(txt)
        f.close()

        # next image
        self.current_frame += 1

        # return to first image
        if self.current_frame == len(self.my_images):
            messagebox.showinfo('This is the End', 'The current sequence of frames is ended.')
            self.root.destroy()
            return

        # change image
        self.canvas.itemconfig(self.image_on_canvas, image=self.my_images[self.current_frame])
        self.curr_frame_name.configure(text=self.filenames[self.current_frame])
        self.count.configure(text='%d/%d' % (self.current_frame, len(self.filenames)))
        self.points.clear()

    def clear_points(self):
        self.points.clear()


points = []


if __name__ == '__main__':

    root = Tk()
    root.title('GUI v2')

    folders_names = os.listdir('../../data/datasets/2d_frames_folders/')

    answer = simpledialog.askstring("Frames folder",
                                    "Which frames' folder do you want to use?\n"+'\n'.join(folders_names),
                                    parent=root)

    path = '../../data/datasets/2d_frames_folders/' + answer + '/'

    if answer is not None and answer != '':
        file_names = os.listdir(path)

        exists = os.path.isfile('../../data/targets/' + answer + '-v2.csv')

        if exists:
            df = pd.read_csv('../../data/targets/' + answer + '-v2.csv', sep=';')
            df_names = df['file']
            csv_names_set = set(df_names)
            file_names_set = set(file_names)
            file_names = list(file_names_set.difference(csv_names_set))
        else:
            f = open('../../data/targets/' + answer + '-v2.csv', 'a+')
            f.write('file;p\n')
            f.close()

        raw_images = []
        photo_imgs = []
        for fr in file_names:
            tmp = cv2.resize(cv2.cvtColor(cv2.imread(path + fr), cv2.COLOR_BGR2RGB), (0, 0), fx=2, fy=2)
            raw_images.append(tmp)
            photo_imgs.append(ImageTk.PhotoImage(image=Image.fromarray(tmp)))

        MainWindow(root, raw_images[0].shape[1], raw_images[0].shape[0], raw_images, photo_imgs, file_names, answer)
        root.mainloop()
    else:
        print('You need to choose a folder.')

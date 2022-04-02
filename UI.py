from tkinter import *
from PIL import ImageTk, Image
import os
import app as a
import numpy as np
import skimage.color as skic
import doctest

class Window :
    """ A class to represent the window

    Attributes
    ----------
    app : tkinter.Tk
        interface of the application
    iteration : int
        step iterator of the application associated with the index of the pages of the application

    Methods
    -------

    """

    def __init__ (self, app, iteration) :
        """ Constructor of all the necessary attributes for the Window object

        Parameters
        ----------
        app : tkinter.Tk
            interface of the application
        iteration : int
            step iterator of the application associated with the index of the pages of the application

        Raises
        ------
        AttributeError
            The ``Raises`` section is a list of all exceptions that are relevant to the interface.
        ValueError
            If `iteration` is not an integer or not positive

        Returns
        -------
        None

        Unitary Test
        ------------
        >>> from tkinter import *
        >>> app = Tk()
        >>> w = Window(app, 0.5)
        Traceback (most recent call last):
        ...
        ValueError: iteration must be an integer
        >>> w = Window(app, -2)
        Traceback (most recent call last):
        ...
        ValueError: iteration must be positive
        >>> app.destroy()
        """
        self.app = app
        if type(iteration) != int :
            raise ValueError("iteration must be an integer")
        elif iteration < 0 :
            raise ValueError("iteration must be positive")
        else :
            self.iteration = iteration

        self.screen_width = self.app.winfo_screenwidth()
        self.screen_height = self.app.winfo_screenheight()

        self.images_data=[]
        self.createDB(a.faces,self.iteration)
        
        self.choice = []
        self.choice[:] = []

        self.choice_carac = []
        self.choice_carac[:] = []

        self.app.title("Intelligent Facial Recognition")
        self.app.geometry(str(self.screen_width)+'x'+str(self.screen_height))
        self.FrameTitle = Frame(self.app, pady = 10)
        self.FrameTitle.pack(fill = BOTH, expand = TRUE)
        Label(self.FrameTitle, text = "WELCOME TO THE FACIAL RECOGNITION SOFTWARE", font = ('Times New Roman', 30, 'bold'), pady = 10).place(anchor = CENTER, relx=.5, rely=.5)

        if self.iteration == 0 :
            self.first_page()
        else :
            self.second_page()

    def first_page(self) :
        """ A function to create the first page of the interface

        This function is called only when the `iteration` parameter is equal to 0.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.FrameText = Frame(self.app, pady = 10)
        self.FrameText.pack(fill = BOTH, expand = TRUE)
        Label(self.FrameText, text = '''Please, choose some basic attributes of the agressor's face between the ones available below
        in order for us to be able to prepare a first sample of faces. Then, press the button "Next step" when you've finished.''', font = ('Times New Roman', 20, 'italic'), pady = 10).place(anchor = CENTER, relx=.5, rely=.5)

        self.FrameAttributes = Frame(self.app, padx = int(0.33*self.screen_width), pady = 10)
        self.FrameAttributes.pack(fill = BOTH, expand = TRUE)
        self.checkbuttons_attrib = []
        k = 0
        self.attributes = ["Fair_Hair","Dark_Hair","Bald","Mustache","Beard","Beardless","Pale_Skin","Intermediate_Skin","Dark_Skin"]
        Label(self.FrameAttributes, text = "CHARACTERISTICS", font = ('Times New Roman', 24, 'italic')).grid(row = 0, column = 1, padx = 10, pady = 10)
        Label(self.FrameAttributes, text = "HAIR", font = ('Times New Roman', 22, 'bold')).grid(row = 1, column = 0, padx = 10, pady = 10)
        Label(self.FrameAttributes, text = "FACIAL HAIR", font = ('Times New Roman', 22, 'bold')).grid(row = 1, column = 1, padx = 10, pady = 10)
        Label(self.FrameAttributes, text = "SKIN", font = ('Times New Roman', 22, 'bold')).grid(row = 1, column = 2, padx = 10, pady = 10)
        for i in range (3) : 
            self.checkbuttons_attrib.append([])
            for j in range(3) :
                self.checkbuttons_attrib[i].append(IntVar())
                Checkbutton(self.FrameAttributes, text = self.attributes[j+k], font = ('Times New Roman', 20), command = self.appearance_btn, compound = 'top', variable = self.checkbuttons_attrib[i][j]).grid(row = j+2, column = i, padx = 10, pady = 10)
            k += 3
        
        self.FrameVal = Frame(app, padx = 10, pady = 10)
        self.FrameVal.pack(fill = BOTH, expand = TRUE)
        self.btn_attrib = Button(self.FrameVal, text = "Next Step", font = ('Times New Roman', 30, 'bold'), state = DISABLED, command = self.carac_choice, relief = GROOVE, width = 10, height = 2)
        self.btn_attrib.place(anchor = CENTER, relx=.5, rely=.5)

    def second_page(self) :
        """ A function to create the pages of the main steps of our software

        The main steps are the ones where the victim is choosing the closest faces from his/her agressor 
        and the software is creating new faces based on what the victim chose.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.FrameText = Frame(self.app, pady = 10)
        self.FrameText.pack(fill = BOTH, expand = TRUE)
        Label(self.FrameText, text = '''You can see 20 faces in front of you and you have to pick two of them that are 
        the closest ones from your agressor. When you are sure about your choice, please press the "Validate"
        button in order to go to the next step !''', font = ('Times New Roman', 20, 'italic')).place(anchor = CENTER, relx=.5, rely=.5)

        self.FrameFace = Frame(self.app, relief = GROOVE, borderwidth = 5, padx = int(0.27*self.screen_width), background = 'beige', pady = 10)
        self.FrameFace.pack(fill = BOTH, expand = TRUE)

        self.database_images = []
        for i in range (20) :
            path = './database'+str(self.iteration)+'/face'+str(i)+'.png'
            face = Image.open(path)
            face = face.resize((int(0.07*self.screen_width), int(0.10*self.screen_height)), Image.ANTIALIAS)
            face_img = ImageTk.PhotoImage(face)
            self.database_images.append(face_img)

        self.checkbuttons = []
        k = 0
        for i in range (5) :
            for j in range (4) : 
                self.checkbuttons.append(IntVar())
                Checkbutton(self.FrameFace, text = "Face n°"+str(k+j+1), font = ('Times New Roman', 12), image = self.database_images[j+k], command = self.appearance_btn, compound = 'top', variable = self.checkbuttons[k+j]).grid(row=j, column=i)
            k += 4

        self.FrameVal = Frame(app, padx = 10, pady = 10)
        self.FrameVal.pack(fill = BOTH, expand = TRUE)
        self.btn = Button(self.FrameVal, text = "Next", font = ('Times New Roman', 20, 'bold'), state = DISABLED, command = self.face_choice, relief = GROOVE, width = 10, height = 2)
        self.btn.place(anchor = CENTER, relx=.5, rely=.5)

        self.var_scale = IntVar()
        self.scale = Scale(self.FrameVal, label = "Resemblance Scale", relief = GROOVE, font = ('Times New Roman', 20, 'bold'), orient = HORIZONTAL, to = 10, length = 200, variable = self.var_scale)
        self.scale.place(anchor = CENTER, relx = .25, rely = .5)

        self.btn_end = Button(self.FrameVal, text = "Finish", font = ('Times New Roman', 20, 'bold'), state = DISABLED, command = self.end, relief = GROOVE, width = 10, height = 2)
        self.btn_end.place(anchor = CENTER, relx = .75, rely = .5)

    def final_page(self) :
        """ A function to create the final page of the software

        In this page, the victim can see his/her final choice.
        Also, the victim can choose to reset the software and come back to the beginning of the simulation.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.FrameText = Frame(self.app, pady = 10)
        self.FrameText.pack(fill = BOTH, expand = TRUE)
        Label(self.FrameText, text = "You have reached the end of the simulation. Please find below the face you've chosen as the closest from your agressor.", font = ('Times New Roman', 20, 'italic')).place(anchor = CENTER, relx=.5, rely=.5)

        self.FrameFace = Frame(self.app, relief = GROOVE, borderwidth = 5, padx = int(0.27*self.screen_width), background = 'beige', pady = 10)
        self.FrameFace.pack(fill = BOTH, expand = TRUE)

        self.FrameVal = Frame(app, padx = 10, pady = 10)
        self.FrameVal.pack(fill = BOTH, expand = TRUE)
        self.btn_final = Button(self.FrameVal, text = "Reset", font = ('Times New Roman', 20, 'bold'), command = self.reset, relief = GROOVE, width = 10, height = 2)
        self.btn_final.pack()

    def carac_choice(self) :
        """ This function is saving and sending the attributes's choice of the victim to the 'app.py' file

        The hair, facial-hair and skin attributes are used to pre-sample the CelebA database of faces.
        This pre-sample will then be sent to the software to propose faces to the victim.
        This function is called when the `btn_attrib` button is pressed.

        Parameters
        ----------
        Nones

        Returns
        -------
        None
        """
        y = 0
        for type in self.checkbuttons_attrib :
            x = 0
            for carac in type :
                if carac.get() == 1 :
                    self.choice_carac.append(self.attributes[x+y])
                x += 1
            y += 3
        self.FrameAttributes.pack_forget()
        self.FrameText.pack_forget()
        self.FrameVal.pack_forget()
        self.second_page()

    def appearance_btn(self) :
        """ This function allows the `btn_attrib` and the `btn_end` buttons to be available or not for the user to press

        The `btn_attrib` button of the first page of the interface is enabled only if the user has selected one attribute 
        per type of characteristic which are the hairs, the facial-hairs and the skin.
        The `btn_end` button of the page of the main steps is enabled if the user has chosen at least one face for the software to compute.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.iteration == 0 :
            states_attrib = []
            for i in range(3) :
                states_attrib.append([var.get() for var in self.checkbuttons_attrib[i]])
            if sum(states_attrib[0]) == 1 and sum(states_attrib[1]) == 1 and sum(states_attrib[2]) == 1 :
                self.btn_attrib['state'] = NORMAL
            else :
                self.btn_attrib['state'] = DISABLED
        else : 
            states_face = [var.get() for var in self.checkbuttons]
            if sum(states_face) >= 1 :
                self.btn['state'] = NORMAL
                if sum(states_face) == 1 :
                    self.btn_end['state'] = NORMAL
                else :
                    self.btn_end['state'] = DISABLED
            elif sum(states_face) == 1 :
                self.btn_end['state'] = NORMAL
            else :
                self.btn['state'] = DISABLED
                self.btn_end['state'] = DISABLED

    def end(self) :
        """ This function opens the last page and saves the final choice of the victim

        It is called when the `btn_end` button is pressed by the user.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        i = 0
        for var in self.checkbuttons :
            if var.get() == 1 :
                self.choice.append(i)
            i += 1
        self.FrameFace.pack_forget()
        self.FrameText.pack_forget()
        self.FrameVal.pack_forget()
        self.final_page()

    def reset(self) :
        """ This function allows the user to reset the simulation and to start again from the beginning

        It is called when the `btn_final` button is pressed by the user.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.FrameTitle.pack_forget()
        self.FrameFace.pack_forget()
        self.FrameText.pack_forget()
        self.FrameVal.pack_forget()
        self.__init__(self.app, 0)

    def face_choice(self) :
        """ This function is the main function of the software compilation

        It is called when the `btn` button is pressed.
        The faces chosen by the user are saved in an array and transferred to the `saveChoices` function.
        Then, the `runApp` function of the `app.py` file is called to create new images with the auto-encoder and the genetic algorithm 
        and a new page of face's choice is presented to the user.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        i = 0
        for var in self.checkbuttons :
            if var.get() == 1 :
                self.choice.append(i)
            i += 1
        self.FrameFace.pack_forget()
        self.FrameText.pack_forget()
        self.FrameTitle.pack_forget()
        self.FrameVal.pack_forget()
        self.saveChoices(self.choice,self.iteration)
        im_choices=[]
        for c in self.choice:
            im_choices.append(a.faces[c])
        im_choices=np.array(im_choices)
        a.runApp(im_choices)
        self.__init__(self.app, self.iteration+1)

    def createDB(self,database,iteration):
        """ This function creates a database of faces in a folder

        The name of the folder is a concatenation of 'database' and the iteration of the software.

        Parameters
        ----------
        database : array
            an array of the faces
        iteration : int
            step iterator of the application associated with the index of the pages of the application

        Returns
        -------
        None
        """
        os.system('mkdir database'+str(iteration))
        i=0
        for f in database:
            data=skic.gray2rgb(f)*255
            data=np.reshape(data,(64,64,3))
            data = data.astype(np.uint8)
            data=Image.fromarray(data)
            if data.mode != 'RGB':
                data = data.convert('RGB')
            self.images_data.append(data)
            data.save("./database"+str(iteration)+"/face"+str(i)+".png")
            i+=1

    def saveChoices(self,choices,iteration):
        """ This function saves the faces' choice of the user in a new folder

        It can be one or several faces. 
        It is called by the `face_choice` function.

        Parameters
        ----------
        choices : array
            an array of indexes of the images
        iteration : int
            step iterator of the application associated with the index of the pages of the application

        Returns
        -------
        None
        """
        os.system('mkdir choice'+str(iteration))
        im_choices=[]
        for c in choices:
            im_choices.append(self.images_data[c+1])
        i=0
        for f in im_choices:
            f.save("./choice"+str(iteration)+"/face"+str(choices[i]+1)+".png")
            i+=1

if __name__ == '__main__' :
    doctest.testmod(verbose = True)
    app = Tk()
    window = Window(app, 0)
    app.mainloop()

from tkinter import *
from tkinter.scrolledtext import ScrolledText
from PIL import ImageTk, Image
import os
import app as a
import numpy as np
import doctest
import shutil

class Window :
    """ A class to represent the window

    Attributes
    ----------
    app : tkinter.Tk
        interface of the application
    iteration : int
        step iterator of the application associated with the index of the pages of the application
    screen_width : int
        width of the screen of the computer launching the user interface
    screen_height : int
        height of the screen of the computer launching the user interface
    images_data : array
        array containing all the images of the current database 
    database_images : array
        array containing all the images of the current database with the right format for the checkbuttons
    choice : array
        array containing the indexes of the images chosen by the user
    choice_carac : array
        array containing the three first attributes chosen by the user
    FrameTitle : tkinter.Frame
        Frame of the title of the curent page
    FrameText : tkinter.Frame
        Frame of the text and instructions of the current page
    FrameAttributes : tkinter.Frame
        Frame of the attribute's selection of the first page
    FrameFace : tkinter.Frame
        Frame of the face's selection of the second page
    FrameVal : tkinter.Frame
        Frame containing all the buttons related to the validation of the current step
    checkbuttons_attrib : array
        array containing the variables of the checkbuttons related to the attributes of the first page
    checkbuttons : array
        array containing the variables of the checkbuttons related to the faces of the second page
    attributes : array
        array containing the name of the attributes of the pre-selection of faces
    btn_attrib : tkinter.Button
        button to validate the first page and step of the software
    btn : tkinter.Button
        button to validate the steps of the second page and run the algorithm to have new faces
    btn_end : tkinter.Button
        button to go the end and last page
    var_scale : tkinter.IntVar
        variable of type integer related to the resemblance sclae bar
    face : tkinter.ImageTk.PhotoImage
        image of the face which represents the final choice of the user

    Methods
    -------
    __init__(app, iteration):
        Constructor of all the necessary attributes for the Window object
    first_page():
        A function to create the first page of the interface
    second_page():
        A function to create the pages of the main steps of our software
    final_page():
        A function to create the final page of the software
    carac_choice():
        This function is saving and sending the attributes's choice of the victim to the 'app.py' file
    appearance_btn():
        This function allows the `btn_attrib` and the `btn_end` buttons to be available or not for the user to press
    end():
        This function opens the last page and saves the final choice of the victim
    reset():
        This function allows the user to reset the simulation and to start again from the beginning
    face_choice():
        This function is the main function of the software compilation
    createDB(database, iteration):
        This function creates a database of faces in a folder
    saveChoices(choices, iteration):
        This function saves the faces' choice of the user in a new folder
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
            If `iteration` is not positive
        TypeError
            If `iteration` is not an integer

        Returns
        -------
        None

        Unitary Test
        ------------
        >>> from tkinter import *
        >>> import app as a
        >>> app = Tk()
        >>> w = Window(app, 0.5)
        Traceback (most recent call last):
        ...
        TypeError: iteration must be an integer
        >>> w = Window(app, -2)
        Traceback (most recent call last):
        ...
        ValueError: iteration must be positive
        >>> app.destroy()
        """
        self.app = app
        if type(iteration) != int :
            raise TypeError("iteration must be an integer")
        elif iteration < 0 :
            raise ValueError("iteration must be positive")
        else :
            self.iteration = iteration

        self.screen_width = self.app.winfo_screenwidth()
        self.screen_height = self.app.winfo_screenheight()

        self.images_data = []
        self.images_data[:] = []
        
        self.choice = []
        self.choice[:] = []

        self.choice_carac = []
        self.choice_carac[:] = []

        self.app.title("LOCAL Software")
        self.app.geometry(str(self.screen_width)+'x'+str(self.screen_height))

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
        self.FrameTitle = Frame(self.app, pady = 10, height = int(0.05*self.screen_height))
        self.FrameTitle.pack(fill = BOTH, expand = TRUE)
        Label(self.FrameTitle, text = "WELCOME TO THE LOCAL SOFTWARE", font = ('Times New Roman', int(0.025*self.screen_width), 'bold'), pady = 10).place(anchor = CENTER, relx=.5, rely=.5)

        self.FrameText = Frame(self.app, pady = 10, height = int(0.05*self.screen_height))
        self.FrameText.pack(fill = BOTH, expand = TRUE)
        Label(self.FrameText, text = '''Please, choose some basic attributes of the agressor's face between the ones available below in order for us to be able to prepare a first sample of faces. 
        Note that you have to select one attribute per column in order to be able to go the next step.
        Then, press the button "Next step" when you've finished.''', font = ('Times New Roman', int(0.015*self.screen_width), 'italic'), pady = 10).place(anchor = CENTER, relx=.5, rely=.5)

        self.FrameAttributes = Frame(self.app, relief = GROOVE, borderwidth = 5, padx = int(0.3*self.screen_width), pady = 50, background = "dark gray")
        self.FrameAttributes.pack(fill = BOTH, expand = TRUE)
        self.checkbuttons_attrib = []
        self.checkbuttons_attrib[:] = []
        k = 0
        self.attributes = ["Fair Hair","Dark Hair","Bald","Mustache","Beard","Beardless","Pale Skin","Intermediate Skin","Dark Skin"]
        Label(self.FrameAttributes, background = "dark gray", text = "CHARACTERISTICS", font = ('Times New Roman', int(0.020*self.screen_width), 'bold')).grid(row = 0, column = 1, padx = 10, pady = 10)
        Label(self.FrameAttributes, background = "dark gray", text = "HAIR", font = ('Times New Roman', int(0.015*self.screen_width), 'bold')).grid(row = 1, column = 0, padx = 10, pady = 10)
        Label(self.FrameAttributes, background = "dark gray", text = "FACIAL HAIR", font = ('Times New Roman', int(0.015*self.screen_width), 'bold')).grid(row = 1, column = 1, padx = 10, pady = 10)
        Label(self.FrameAttributes, background = "dark gray", text = "SKIN", font = ('Times New Roman', int(0.015*self.screen_width), 'bold')).grid(row = 1, column = 2, padx = 10, pady = 10)
        for i in range (3) : 
            self.checkbuttons_attrib.append([])
            for j in range(3) :
                self.checkbuttons_attrib[i].append(IntVar())
                Checkbutton(self.FrameAttributes, background = "dark gray", text = self.attributes[j+k], font = ('Times New Roman', int(0.015*self.screen_width)), command = self.appearance_btn, compound = 'top', variable = self.checkbuttons_attrib[i][j]).grid(row = j+2, column = i, padx = 10, pady = 10)
            k += 3
        
        self.FrameVal = Frame(self.app, padx = 10, pady = 10, height = int(0.06*self.screen_height))
        self.FrameVal.pack(fill = BOTH, expand = TRUE)
        self.btn_attrib = Button(self.FrameVal, text = "Next Step", font = ('Times New Roman', int(0.020*self.screen_width), 'bold'), state = DISABLED, command = self.carac_choice, relief = GROOVE, width = 10, height = 2)
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
        self.FrameTitle = Frame(self.app, pady = 10, height = int(0.05*self.screen_height))
        self.FrameTitle.pack(fill = BOTH, expand = TRUE)
        Label(self.FrameTitle, text = "FINDING THE RIGHT FACE", font = ('Times New Roman', int(0.025*self.screen_width), 'bold'), pady = 10).place(anchor = CENTER, relx=.5, rely=.5)

        self.FrameText = Frame(self.app, pady = 10, height = int(0.08*self.screen_height))
        self.FrameText.pack(fill = X, expand = TRUE)
        st = ScrolledText(self.FrameText, width = self.screen_width, height = int(0.006*self.screen_height), font = ('Times New Roman', int(0.015*self.screen_width), 'italic'))
        st.insert(INSERT, '''----------INSTRUCTIONS----------
        \n
        You can see 20 faces in front of you and you have to select the closest ones from your agressor's face.
        Note that you have to pick at least one face to have a new sample of faces in order to run or rerun the algorithm. 
        \n
        Also you can move the "Resemblance scale" between 0 and 10 if you want to give more informations to the algorithm. 
        Be aware that 0 is a really far from the agressor's face and 10 is really close from it.
        Then press the "Next Step" button if you want to find a closer face.
        \n
        When you are sure about your final choice, please press the "Finish" button in order to finish the simulation. 
        Note that you can only select one face to activate this button.''', "text")
        st.configure(state = DISABLED)
        st.tag_configure("text", justify = CENTER)
        st.tag_add("text", INSERT)
        st.pack(fill = X, side = TOP)

        self.FrameFace = Frame(self.app, relief = GROOVE, borderwidth = 5, padx = int(0.27*self.screen_width), background = 'beige', pady = 10)
        self.FrameFace.pack(fill = BOTH, expand = TRUE)

        self.createDB(a.faces,self.iteration)
        self.database_images = []
        self.database_images[:] = []
        for i in range (20) :
            path = './database'+str(self.iteration)+'/face'+str(i)+'.png'
            face = Image.open(path)
            face = face.resize((int(0.07*self.screen_width), int(0.10*self.screen_height)), Image.ANTIALIAS)
            face_img = ImageTk.PhotoImage(face)
            self.database_images.append(face_img)

        self.checkbuttons = []
        self.checkbuttons[:] = []
        k = 0
        for i in range (5) :
            for j in range (4) : 
                self.checkbuttons.append(IntVar())
                Checkbutton(self.FrameFace, text = "Face n??"+str(k+j+1), font = ('Times New Roman', int(0.010*self.screen_width)), image = self.database_images[j+k], command = self.appearance_btn, compound = 'top', variable = self.checkbuttons[k+j]).grid(row=j, column=i)
            k += 4

        self.FrameVal = Frame(app, padx = 10, pady = 10, height = int(0.08*self.screen_height))
        self.FrameVal.pack(fill = BOTH, expand = TRUE)
        self.btn = Button(self.FrameVal, text = "Next", font = ('Times New Roman', int(0.020*self.screen_width), 'bold'), state = DISABLED, command = self.face_choice, relief = GROOVE, width = 10, height = 2)
        self.btn.place(anchor = CENTER, relx=.5, rely=.5)

        self.var_scale = IntVar()
        scale = Scale(self.FrameVal, label = "Resemblance Scale", relief = GROOVE, borderwidth = 2, font = ('Times New Roman', int(0.012*self.screen_width), 'bold'), orient = HORIZONTAL, to = 10, length = 200, variable = self.var_scale)
        scale.place(anchor = CENTER, relx = .25, rely = .5)

        self.btn_end = Button(self.FrameVal, text = "Finish", font = ('Times New Roman', int(0.020*self.screen_width), 'bold'), state = DISABLED, command = self.end, relief = GROOVE, width = 10, height = 2)
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
        self.FrameTitle = Frame(self.app, pady = 10)
        self.FrameTitle.pack(fill = BOTH, expand = TRUE)
        Label(self.FrameTitle, text = "FINAL RESULT", font = ('Times New Roman', int(0.025*self.screen_width), 'bold'), pady = 10).place(anchor = CENTER, relx=.5, rely=.5)

        self.FrameText = Frame(self.app, pady = 10)
        self.FrameText.pack(fill = BOTH, expand = TRUE)
        Label(self.FrameText, text = "You have reached the end of the simulation. Please find below the face you've chosen as the closest from your agressor.", font = ('Times New Roman', int(0.015*self.screen_width), 'italic')).place(anchor = CENTER, relx=.5, rely=.5)

        self.FrameFace = Frame(self.app, relief = GROOVE, borderwidth = 5, padx = int(0.27*self.screen_width), pady = 10, background = 'beige', height = int(0.4*self.screen_height))
        self.FrameFace.pack(fill = BOTH, expand = TRUE)
        path = './database'+str(self.iteration)+'/face'+str(self.choice[0])+'.png'
        face = Image.open(path)
        face = face.resize((int(0.2*self.screen_width), int(0.3*self.screen_height)), Image.ANTIALIAS)
        self.face = ImageTk.PhotoImage(face)
        canvas = Canvas(self.FrameFace, width=0.21*self.screen_width, height=0.31*self.screen_height)
        canvas.create_image(10, 10, image = self.face, anchor = NW)
        canvas.place(anchor = CENTER, relx=.5, rely=.5)

        self.FrameVal = Frame(app, padx = 10, pady = 10)
        self.FrameVal.pack(fill = BOTH, expand = TRUE)
        btn_reset = Button(self.FrameVal, text = "Reset", font = ('Times New Roman', int(0.020*self.screen_width), 'bold'), command = self.reset, relief = GROOVE, width = 10, height = 2)
        btn_reset.pack()

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
                    self.choice_carac.append(self.attributes[x+y].replace(" ", "_"))
                x += 1
            y += 3
        a.choice(self.choice_carac)
        self.FrameTitle.pack_forget()
        self.FrameAttributes.pack_forget()
        self.FrameText.pack_forget()
        self.FrameVal.pack_forget()
        self.iteration += 1
        self.__init__(self.app, self.iteration)

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
        self.FrameTitle.pack_forget()
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
        cleanDir()
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
        a.runApp(im_choices, self.var_scale)
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

        Raises
        ------
        ValueError
            If `database` is empty
        ValueError
            If `iteration` is not positive
        TypeError
            If `iteration` is not an integer
        TypeError
            If the content of `database` is not numpy arrays

        Returns
        -------
        None

        Unitary Tests
        -------------
        >>> from PIL import Image
        >>> from tkinter import *
        >>> import numpy as np
        >>> import app as a
        >>> app = Tk()
        >>> w = Window(app, 0)
        >>> w.createDB([],0.5)
        Traceback (most recent call last):
        ...
        TypeError: The iteration parameter is not an integer
        >>> w.createDB([],-1)
        Traceback (most recent call last):
        ...
        ValueError: The iteration parameter is not positive
        >>> w.createDB([],0)
        Traceback (most recent call last):
        ...
        ValueError: The database is empty
        >>> w.createDB([1,2,3],0)
        Traceback (most recent call last):
        ...
        TypeError: The content of the database is not an Image
        >>> app.destroy()
        """
        if type(iteration) != int :
            raise TypeError("The iteration parameter is not an integer")
        elif iteration < 0 : 
            raise ValueError('The iteration parameter is not positive')
        else :
            if iteration == 0:
                iteration += 1
            os.system('mkdir database'+str(iteration))
            i = 0
            if len(database) == 0 :
                raise ValueError("The database is empty")
            else :
                for f in database:
                    if not (isinstance(f, np.ndarray)) :
                        raise TypeError("The content of the database is not an Image")
                    else :
                        data=np.reshape(f,(218,178,3))*255
                        data = data.astype(np.uint8)
                        data=Image.fromarray(data)
                        if data.mode != 'RGB':
                            data = data.convert('RGB')
                        self.images_data.append(data)
                        data.save("./database"+str(iteration)+"/face"+str(i)+".png")
                        i += 1
            
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

        Raises
        ------
        ValueError
            If `iteration` is not positive
        TypeError
            If `iteration` is not an integer
        ValueError
            If `choices` is empty
        TypeError
            If the content of `choices` is not a positive integer

        Returns
        -------
        None

        Unitary Tests
        -------------
        >>> from tkinter import *
        >>> import app as a
        >>> app = Tk()
        >>> w = Window(app, 0)
        >>> w.saveChoices([],0.5)
        Traceback (most recent call last):
        ...
        TypeError: The iteration parameter is not an integer
        >>> w.saveChoices([],-1)
        Traceback (most recent call last):
        ...
        ValueError: The iteration parameter is not positive
        >>> w.saveChoices([-0.5],0)
        Traceback (most recent call last):
        ...
        TypeError: c is not positive or not an integer
        >>> w.saveChoices([],0)
        Traceback (most recent call last):
        ...
        ValueError: choices is empty
        >>> app.destroy()
        """
        if type(iteration) != int :
            raise TypeError("The iteration parameter is not an integer")
        elif iteration < 0 : 
            raise ValueError('The iteration parameter is not positive')
        else :
            os.system('mkdir choice'+str(iteration+1))
            im_choices=[]
            if len(choices) == 0 :
                raise ValueError("choices is empty")
            else :
                for c in choices:
                    if type(c) != int and c < 0 :
                        raise TypeError("c is not positive or not an integer")
                    im_choices.append(self.images_data[c+1])
                i=0
                for f in im_choices:
                    f.save("./choice"+str(iteration+1)+"/face"+str(choices[i]+1)+".png")
                    i+=1
def cleanDir() :
    """ This function is useful to reset the software
    
    It erases the databases folders and their content in order to reset the software and run it again.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    for (dirpath, dirnames, filenames) in os.walk("../Projet4BIM") :
            if dirpath == "../Projet4BIM" :
                for d in dirnames :
                    if d[0:8] == "database" or d[0:6] == "choice" :
                        shutil.rmtree(d)

if __name__ == '__main__' :
    doctest.testmod(verbose = True)
    app = Tk()
    window = Window(app, 0)
    app.mainloop()

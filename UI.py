from tkinter import *
from PIL import ImageTk, Image
import os
import app as a
import numpy as np
import skimage.color as skic

class FirstStep :

    def __init__ (self, app, iteration) :
        self.app = app
        self.iteration = iteration
        self.app.title("Intelligent Facial Recognition")

        self.screen_width = self.app.winfo_screenwidth()
        self.screen_height = self.app.winfo_screenheight()

        self.images_data=[]
        self.createDB(a.faces,self.iteration)
        

        self.choice = []

        self.app.geometry(str(self.screen_width)+'x'+str(self.screen_height))

        self.FrameTitle = Frame(self.app, pady = 10)
        self.FrameTitle.pack(fill = BOTH, expand = TRUE)
        Label(self.FrameTitle, text = "WELCOME TO THE FACIAL RECOGNITION SOFTWARE", font = ('Times New Roman', 30, 'bold'), pady = 10).place(anchor = CENTER, relx=.5, rely=.5)
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
        self.btn = Button(self.FrameVal, text = "Validate", font = ('Times New Roman', 20, 'bold'), state = DISABLED, command = self.face_choice, relief = GROOVE, width = 10, height = 2)
        self.btn.place(anchor = CENTER, relx=.5, rely=.5)

    def appearance_btn(self) :
        states = [var.get() for var in self.checkbuttons]
        if sum(states) == 2 :
            self.btn['state'] = NORMAL
        else :
            self.btn['state'] = DISABLED

    def face_choice(self) :
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
        os.system('mkdir choice'+str(iteration))
        im_choices=[]
        for c in choices:
            im_choices.append(self.images_data[c+1])
        i=0
        for f in im_choices:
            f.save("./choice"+str(iteration)+"/face"+str(choices[i]+1)+".png")
            i+=1

app = Tk()
window = FirstStep(app, 0)
app.mainloop()

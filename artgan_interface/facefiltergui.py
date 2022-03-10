import PySimpleGUI as sg
import os.path
import facefiltertest

def main():
    sg.theme("Dark")
    file_list_column = [
        [
            sg.Text("Image Folder"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
        ],
        [
            sg.Listbox(
                values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
            )
        ],
    ]

    # For now will only show the name of the file that was chosen
    input_image_viewer_column = [
        [sg.Text("Select an input image from the left")],
        [sg.Text(size=(40, 1), key="-TOUT-")],
        [sg.Image(key="-INPUT_IMAGE-")],
    ]

    output_image_viewer_column = [
        [sg.Text("Your translated image!")],
        [sg.Text(size=(40, 1), key="-TOUT-")],
        [sg.Image(key="-OUTPUT_IMAGE-")],
    ]

    # ----- Full layout -----
    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(input_image_viewer_column),
            sg.Column(output_image_viewer_column),
            
        ],
        [sg.Text("Select an artGAN:")],
        [sg.Radio("CUBISM", "Radio", size=(10, 1), key="-CUBISM-")],
        [sg.Radio("POP ART", "Radio", size=(10, 1), key="-POPART-")],
        [sg.Button("Generate!", size=(10, 1))],
    ]

    window = sg.Window("artGAN", layout)

    # Run the Event Loop
    filename = None
    folder = None
    fakefile = "fake-image.png"
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        # Folder name was filled in, make a list of files in the folder
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                # Get list of files in folder
                file_list = os.listdir(folder)
            except:
                file_list = []

            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith(".png")
            ]
            window["-FILE LIST-"].update(fnames)
        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                window["-TOUT-"].update(filename)
                window["-INPUT_IMAGE-"].update(filename=filename)
                

            except:
                pass
        if event == "Generate!":
            if values["-CUBISM-"]:
                    pretrained_file = 'cycleGAN_256_cubism.pth'
                    fake = facefiltertest.generateImage(filename, pretrained_file)
                    facefiltertest.save_tensor_images(fake)
                    window["-TOUT-"].update(fakefile)
                    window["-OUTPUT_IMAGE-"].update(filename=fakefile)
            elif values["-POPART-"]:
                    pretrained_file = 'cycleGAN_256_popart_4.pth'
                    fake = facefiltertest.generateImage(filename, pretrained_file)
                    facefiltertest.save_tensor_images(fake)
                    window["-TOUT-"].update(fakefile)
                    window["-OUTPUT_IMAGE-"].update(filename=fakefile)
            

    window.close()

main()
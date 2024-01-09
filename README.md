# Urdu-KeyWord-Spotting-API
Urdu Keyword Spotting API
Overview

This project houses an Urdu Keyword Spotting API developed using Flask, designed to identify specific keywords within audio data. The API is seamlessly integrated with an interactive web interface, allowing users to upload haystack and needle audio files. The UI provides a user-friendly experience, visualizing the positions of keywords within the haystack and facilitating audio playback of the relevant sections.
Project Structure

    app.py: The Flask application that serves as the backend for the Urdu Keyword Spotting API.
    templates/index.html: The HTML template for the interactive web interface.
    static: 
        img/: Folder containing images used within the interface.
    Audio: Contains sample audios that you can use as needle and hastack
    needle -> keyword to search
    Haystack -> audio file to search needle in.

        
*The pretrained.py file has the model for spotting the Urdu Keywords already trained incase you need to use it*


![API](https://github.com/zartaj-asim/Urdu-KeyWord-Spotting-API/assets/109308812/d9991735-0e5b-4e3d-bfdf-61e405f9313a)

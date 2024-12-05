import os
from time import sleep
from django.conf import settings
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .model import inference

def main(request):

    prediction = {}
    current_file = None

    if request.method == 'POST':
        # Handle file uploads
        if request.FILES.get('image'):
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage()
            fs.save(uploaded_file.name, uploaded_file)
            return redirect('main')

        # Handle predictions
        selected_file = request.POST.get('selected_file')
        if selected_file:
            current_file = selected_file
            prediction = inference(os.path.join(settings.MEDIA_ROOT, os.path.basename(selected_file)))

    # File browsing
    file_urls = []
    media_root = settings.MEDIA_ROOT
    if os.path.exists(media_root):
        for filename in os.listdir(media_root):
            file_urls.append({
                'name': filename,
                'url': settings.MEDIA_URL + filename,
            })

    return render(request, 'main.html', {'files': file_urls, 'prediction': prediction, 'current_file':current_file})

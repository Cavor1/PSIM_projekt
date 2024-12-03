import os
from django.conf import settings
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage

def main(request):
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
            print(f"Selected file for prediction: {selected_file}")
            # Add your model prediction logic here

    # File browsing
    file_urls = []
    media_root = settings.MEDIA_ROOT
    if os.path.exists(media_root):
        for filename in os.listdir(media_root):
            file_urls.append({
                'name': filename,
                'url': settings.MEDIA_URL + filename,
            })

    return render(request, 'main.html', {'files': file_urls})

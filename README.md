# Portfolio Chatbot

## Frontend
- Folder ini berisi file website statis (HTML, CSS, JS).
- Bisa di-deploy ke Netlify, GitHub Pages, dsb.

## Backend
- Folder ini berisi source code chatbot Python (Flask).
- Jalankan backend secara lokal atau deploy ke Render/Railway/Heroku.

## Cara Menjalankan Lokal
1. Jalankan backend:
   ```
   cd backend
   pip install -r requirements.txt
   python app.py
   ```
2. Jalankan frontend:
   ```
   cd frontend
   python -m http.server 8000
   ```
3. Buka browser ke http://localhost:8000

## Deploy
- Frontend bisa diupload ke Netlify/GitHub Pages.
- Backend harus dihosting di layanan Python (Render, Railway, Heroku, dsb).
- Ubah URL fetch di JS agar mengarah ke backend online.

# Bridge Dashboard - Setup & Troubleshooting

## Problem: "Error loading CSV file: Failed to fetch"

This happens because modern browsers block **CORS (Cross-Origin)** requests when opening HTML files directly using the `file://` protocol for security reasons.

## ✅ Solution 1: File Upload (Simplest)

1. Open `bridge_criticality_interactive_dashboard.html` in your browser
2. Click **"📁 Load CSV File"** button
3. Select your `element_criticality_forecast_2025_2029.csv` file
4. Dashboard will load immediately

**Pros**: Works everywhere, no setup needed  
**Cons**: Need to select file each time

---

## ✅ Solution 2: Local HTTP Server (Recommended)

### On Windows (PowerShell):
```powershell
cd C:\Users\clnnaLAB\Desktop\AutoDM\AutoDM
python serve_dashboard.py
```

Then open `http://localhost:8000` in your browser.

### or use built-in Python:
```powershell
cd C:\Users\clnnaLAB\Desktop\AutoDM\AutoDM
python -m http.server 8000
```

Then open `http://localhost:8000/bridge_criticality_interactive_dashboard.html`

### On Mac/Linux:
```bash
cd ~/Desktop/AutoDM/AutoDM
python3 -m http.server 8000
```

Then open `http://localhost:8000/bridge_criticality_interactive_dashboard.html`

**Pros**: 
- CSV loads automatically
- Persists across page reloads
- Professional setup

**Cons**: 
- Requires running a server

---

## ✅ Solution 3: Browser Extensions

Some browsers/extensions allow local file CORS:
- **Chrome**: Use Web Server for Chrome extension
- **Firefox**: Disable CORS for local files in `about:config`

---

## Troubleshooting

### "Failed to fetch" but file upload selected
- Check that the CSV file exists in the same folder as HTML
- Verify filename matches: `element_criticality_forecast_2025_2029.csv`
- Try re-selecting the file

### Port 8000 already in use
```powershell
# Try a different port:
python -m http.server 8001
# Then open: http://localhost:8001/...
```

### CSV loading but no bridges appear
- Verify CSV has the required column: `STRUCTURE_NUMBER`
- Check CSV is not corrupted (open in Excel/text editor)
- Look at browser console (F12) for JavaScript errors

### Charts don't render
- Ensure Plotly.js loads (check network tab in F12)
- Try refreshing page (Ctrl+R)
- Try a different browser

---

## File Locations

```
AutoDM/
├── bridge_criticality_interactive_dashboard.html  ← Open this in browser
├── serve_dashboard.py                             ← Run this for server mode
├── DASHBOARD_SETUP.md                             ← This file
├── element_criticality_forecast_2025_2029.csv     ← Put your CSV here
├── bridge_criticality_summary_2025_2029.csv       ← Optional
└── notebooks/
    └── forecasting_arranged.ipynb                 ← Generates CSVs
```

---

## Recommended Workflow

1. **Generate CSVs from Jupyter notebook**
   ```
   forecasting_arranged.ipynb → element_criticality_forecast_2025_2029.csv
   ```

2. **Move CSV to same folder as dashboard**
   ```
   cp element_criticality_forecast_2025_2029.csv AutoDM/
   ```

3. **Start local server**
   ```
   cd AutoDM/
   python serve_dashboard.py
   ```

4. **Open dashboard in browser**
   - Browser opens automatically
   - Or go to: http://localhost:8000

5. **Explore and export**
   - Select bridges from dropdown
   - Download charts as PNG
   - Export results

---

## Questions?

See `CRITICALITY_DASHBOARD_README.md` for full documentation on metrics, interpretation, and analysis.

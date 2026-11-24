# AI Spam Mail Detection — Local Server Launcher
# Launches the Python Flask ML backend & web UI on http://localhost:3000

$root = "c:\Users\ADMIN\Desktop\mini_project"
$port = 3000

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  AI SPAM MAIL DETECTOR — Server Launcher" -ForegroundColor Green
Write-Host "  Machine Learning & NLP Threat Intelligence Core" -ForegroundColor Gray
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Check if python is available to run full ML backend
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue

if ($pythonCmd) {
    Write-Host "  Python detected: Starting Flask ML Backend on http://localhost:$port/" -ForegroundColor Green
    Write-Host "  Model: Linear SVM (Calibrated) with TF-IDF N-Grams" -ForegroundColor Gray
    Write-Host "  Press Ctrl+C to stop.`n" -ForegroundColor Gray
    Set-Location $root
    python app.py
    exit
}

# Fallback: pure PowerShell HTTP listener (static files only)
Write-Host "  Python not found in PATH. Falling back to PowerShell HTTP Listener..." -ForegroundColor Yellow

$mimeTypes = @{
    '.html' = 'text/html; charset=utf-8'
    '.css'  = 'text/css; charset=utf-8'
    '.js'   = 'application/javascript; charset=utf-8'
    '.png'  = 'image/png'
    '.jpg'  = 'image/jpeg'
    '.svg'  = 'image/svg+xml'
    '.json' = 'application/json'
    '.txt'  = 'text/plain'
}

$http = [System.Net.HttpListener]::new()
$http.Prefixes.Add("http://localhost:$port/")
$http.Start()

Write-Host "  Static server running at http://localhost:$port/" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop.`n" -ForegroundColor Gray

while ($http.IsListening) {
    $ctx  = $http.GetContext()
    $req  = $ctx.Request
    $resp = $ctx.Response

    $localPath = $req.Url.LocalPath
    if ($localPath -eq '/') { $localPath = '/index.html' }

    $filePath = [System.IO.Path]::GetFullPath(
        [System.IO.Path]::Combine($root, $localPath.TrimStart('/').Replace('/', '\'))
    )

    if (-not $filePath.StartsWith($root)) {
        $resp.StatusCode = 403
        $resp.Close()
        continue
    }

    if (Test-Path $filePath -PathType Leaf) {
        $ext     = [System.IO.Path]::GetExtension($filePath).ToLower()
        $mime    = if ($mimeTypes.ContainsKey($ext)) { $mimeTypes[$ext] } else { 'application/octet-stream' }
        $content = [System.IO.File]::ReadAllBytes($filePath)

        $resp.StatusCode      = 200
        $resp.ContentType     = $mime
        $resp.ContentLength64 = $content.Length
        $resp.OutputStream.Write($content, 0, $content.Length)
    } else {
        $body = [System.Text.Encoding]::UTF8.GetBytes("<h1>404 Not Found</h1><p>$localPath</p>")
        $resp.StatusCode      = 404
        $resp.ContentType     = 'text/html'
        $resp.ContentLength64 = $body.Length
        $resp.OutputStream.Write($body, 0, $body.Length)
    }

    $resp.Close()
}

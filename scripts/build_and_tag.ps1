Param(
    [string]$User = $env:DOCKERHUB_USERNAME,
    [string]$Tag = $env:TAG
)

if (-not $User) {
    Write-Host "Please set DOCKERHUB_USERNAME environment variable or pass -User <username>" -ForegroundColor Yellow
    exit 1
}
if (-not $Tag) { $Tag = "latest" }

$env:DOCKERHUB_USERNAME = $User
$env:TAG = $Tag

Write-Host "Building backend image: $User/mlopsproject-backend:$Tag"
docker-compose build backend
Write-Host "Tagging backend image"
try {
    docker tag "${User}/mlopsproject-backend:latest" "$User/mlopsproject-backend:$Tag" 2>$null
} catch {
    Write-Host "Tag may already exist or tagging failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "Building frontend image: $User/mlopsproject-frontend:$Tag"
docker-compose build frontend
Write-Host "Tagging frontend image"
try {
    docker tag "${User}/mlopsproject-frontend:latest" "$User/mlopsproject-frontend:$Tag" 2>$null
} catch {
    Write-Host "Tag may already exist or tagging failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "Build complete. Run 'scripts\push_images.ps1 -User $User -Tag $Tag' after docker login to push images."
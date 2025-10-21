Param(
    [string]$User = $env:DOCKERHUB_USERNAME,
    [string]$Tag = $env:TAG
)

if (-not $User) {
    Write-Host "Please set DOCKERHUB_USERNAME environment variable or pass -User <username>" -ForegroundColor Yellow
    exit 1
}
if (-not $Tag) { $Tag = "latest" }

# Ensure user is logged into Docker locally; do not accept password in script
Write-Host "Make sure you have run: docker login"

$backendImage = "$User/mlopsproject-backend:$Tag"
$frontendImage = "$User/mlopsproject-frontend:$Tag"

Write-Host "Pushing $backendImage"
docker push $backendImage

Write-Host "Pushing $frontendImage"
docker push $frontendImage

Write-Host "Push complete." -ForegroundColor Green
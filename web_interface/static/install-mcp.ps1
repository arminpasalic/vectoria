# Vectoria MCP Server installer for Windows PowerShell 5.1+
# Install:   irm https://vectoria.app/static/install-mcp.ps1 | iex
# Uninstall: & ([scriptblock]::Create((irm https://vectoria.app/static/install-mcp.ps1))) -Uninstall

param(
    [string]$BaseUrl = "https://vectoria.app",
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$InstallDir = Join-Path $env:USERPROFILE ".vectoria-mcp"

$Clients = @(
    @{ Name = "Claude Desktop"; Path = (Join-Path $env:APPDATA "Claude\claude_desktop_config.json"); Format = "mcp_object" },
    @{ Name = "Cursor"; Path = (Join-Path $env:USERPROFILE ".cursor\mcp.json"); Format = "mcp_object" },
    @{ Name = "OpenCode"; Path = (Join-Path $env:USERPROFILE ".config\opencode\config.json"); Format = "opencode" },
    @{ Name = "Zed"; Path = (Join-Path $env:APPDATA "Zed\settings.json"); Format = "zed" },
    @{ Name = "Continue.dev"; Path = (Join-Path $env:USERPROFILE ".continue\config.json"); Format = "continue" }
)

function Read-Config([string]$Path) {
    if (-not (Test-Path $Path)) { return [pscustomobject]@{} }
    try { return (Get-Content -Raw $Path | ConvertFrom-Json) }
    catch { throw "Cannot update invalid JSON config: $Path`n$($_.Exception.Message)" }
}

function Set-Property($Object, [string]$Name, $Value) {
    if ($Object.PSObject.Properties[$Name]) { $Object.$Name = $Value }
    else { $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value }
}

function Save-Config([string]$Path, $Config) {
    $Directory = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force $Directory | Out-Null
    $Config | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 $Path
}

function Add-ClientConfig($Client, [string]$NodeExe, [string]$Entry, [string]$AllowedOrigin) {
    $Config = Read-Config $Client.Path
    $ServerArgs = @($Entry, "--allowed-origin", $AllowedOrigin)
    switch ($Client.Format) {
        "mcp_object" {
            if (-not $Config.mcpServers) { Set-Property $Config "mcpServers" ([pscustomobject]@{}) }
            Set-Property $Config.mcpServers "vectoria" ([pscustomobject]@{ command = $NodeExe; args = $ServerArgs })
        }
        "opencode" {
            if (-not $Config.mcp) { Set-Property $Config "mcp" ([pscustomobject]@{}) }
            Set-Property $Config.mcp "vectoria" ([pscustomobject]@{ type = "local"; command = @($NodeExe) + $ServerArgs })
        }
        "zed" {
            if (-not $Config.context_servers) { Set-Property $Config "context_servers" ([pscustomobject]@{}) }
            Set-Property $Config.context_servers "vectoria" ([pscustomobject]@{
                command = [pscustomobject]@{ path = $NodeExe; args = $ServerArgs }
            })
        }
        "continue" {
            $Servers = @($Config.mcpServers | Where-Object { $_.name -ne "vectoria" })
            $Servers += [pscustomobject]@{ name = "vectoria"; command = $NodeExe; args = $ServerArgs }
            Set-Property $Config "mcpServers" $Servers
        }
    }
    Save-Config $Client.Path $Config
}

function Remove-ClientConfig($Client) {
    if (-not (Test-Path $Client.Path)) { return }
    $Config = Read-Config $Client.Path
    switch ($Client.Format) {
        "mcp_object" { if ($Config.mcpServers) { $Config.mcpServers.PSObject.Properties.Remove("vectoria") } }
        "opencode" { if ($Config.mcp) { $Config.mcp.PSObject.Properties.Remove("vectoria") } }
        "zed" { if ($Config.context_servers) { $Config.context_servers.PSObject.Properties.Remove("vectoria") } }
        "continue" { Set-Property $Config "mcpServers" @($Config.mcpServers | Where-Object { $_.name -ne "vectoria" }) }
    }
    Save-Config $Client.Path $Config
    Write-Host "   Removed from $($Client.Path)" -ForegroundColor Green
}

if ($Uninstall) {
    Write-Host "Uninstalling Vectoria MCP server..." -ForegroundColor Cyan
    if (Test-Path $InstallDir) { Remove-Item -Recurse -Force $InstallDir }
    foreach ($Client in $Clients) { Remove-ClientConfig $Client }
    Write-Host "Uninstall complete. Restart your AI clients." -ForegroundColor Green
    return
}

Write-Host "Installing Vectoria MCP server to $InstallDir" -ForegroundColor Cyan
$Node = Get-Command node.exe -ErrorAction SilentlyContinue
if (-not $Node) { throw "Node.js v18+ was not found. Install it from https://nodejs.org and run this command again." }
$NodeMajor = [int](& $Node.Source -p "process.versions.node.split('.')[0]")
if ($NodeMajor -lt 18) { throw "Node.js v18+ is required. Installed: $(& $Node.Source -v)" }
Write-Host "Node.js $(& $Node.Source -v) ($($Node.Source))" -ForegroundColor Green

New-Item -ItemType Directory -Force (Join-Path $InstallDir "tools") | Out-Null
$Files = @(
    "package.json", "index.js", "bridge.js",
    "tools/search.js", "tools/rag.js", "tools/data.js", "tools/metadata.js",
    "tools/config.js", "tools/dataset.js", "tools/analysis.js", "tools/annotations.js",
    "tools/clusters.js", "tools/metrics.js", "tools/sessions.js"
)
Write-Host "Downloading MCP server files..." -ForegroundColor Cyan
foreach ($File in $Files) {
    $Destination = Join-Path $InstallDir ($File -replace "/", "\")
    Invoke-WebRequest -UseBasicParsing "$($BaseUrl.TrimEnd('/'))/static/mcp-server/$File" -OutFile $Destination
    Write-Host "   downloaded $File"
}

try { Get-Content -Raw (Join-Path $InstallDir "package.json") | ConvertFrom-Json | Out-Null }
catch { throw "The downloaded package.json is invalid. Check that $BaseUrl serves the Vectoria static files." }

Write-Host "Installing Node dependencies..." -ForegroundColor Cyan
Push-Location $InstallDir
try { & npm.cmd install --omit=dev; if ($LASTEXITCODE -ne 0) { throw "npm install failed with exit code $LASTEXITCODE" } }
finally { Pop-Location }

Write-Host "Configuring detected AI clients..." -ForegroundColor Cyan
$Configured = @()
$Skipped = @()
foreach ($Client in $Clients) {
    $Directory = Split-Path -Parent $Client.Path
    if ((Test-Path $Directory) -or (Test-Path $Client.Path)) {
        Add-ClientConfig $Client $Node.Source (Join-Path $InstallDir "index.js") $BaseUrl
        $Configured += $Client.Name
        Write-Host "   configured $($Client.Name)" -ForegroundColor Green
    } else { $Skipped += $Client.Name }
}

Write-Host ""
Write-Host "Vectoria MCP server installed!" -ForegroundColor Green
if ($Configured.Count) { Write-Host "Configured: $($Configured -join ', ')" }
if ($Skipped.Count) { Write-Host "Not found: $($Skipped -join ', ')" }
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Fully quit and reopen your AI client(s)."
Write-Host "2. Open Vectoria in your browser."
Write-Host "3. Advanced Settings -> MCP Bridge -> enable the toggle."
Write-Host "4. Allow Local Network Access if your browser prompts."
Write-Host "5. The AI client launches the local MCP server automatically."

Option Explicit

Dim shell
Dim fso
Dim command
Dim scriptDir
Dim projectRoot
Dim logsDir
Dim logPath
Dim psPath
Dim keepalivePath
Dim logFile

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
projectRoot = fso.GetParentFolderName(scriptDir)
logsDir = fso.BuildPath(projectRoot, "logs")
logPath = fso.BuildPath(logsDir, "voiceclone_launcher.log")
keepalivePath = fso.BuildPath(scriptDir, "keepalive_server.ps1")
psPath = shell.ExpandEnvironmentStrings("%SystemRoot%") & "\System32\WindowsPowerShell\v1.0\powershell.exe"

If Not fso.FolderExists(logsDir) Then
    fso.CreateFolder(logsDir)
End If

Set logFile = fso.OpenTextFile(logPath, 8, True)
logFile.WriteLine Now & " lancement demande via start_voiceclone_server_hidden.vbs"
logFile.WriteLine "Projet: " & projectRoot
logFile.WriteLine "Script: " & keepalivePath
logFile.Close

shell.CurrentDirectory = projectRoot
command = """" & psPath & """ -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & keepalivePath & """"

' 0 = fenetre cachee, False = ne pas attendre la fin du serveur.
shell.Run command, 0, False

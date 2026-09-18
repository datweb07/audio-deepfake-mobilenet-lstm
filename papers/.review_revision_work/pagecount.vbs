Option Explicit
Dim app, doc
Set app = GetObject(, "Word.Application")
Set doc = app.ActiveDocument
doc.Repaginate
WScript.Echo "DOC=" & doc.Name
WScript.Echo "PAGES=" & doc.ComputeStatistics(2)

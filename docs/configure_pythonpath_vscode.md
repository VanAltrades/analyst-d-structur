File structuer looks like this...

C:.
└───analyst-d-structur
    ├───docs
    ├───src
    │   └───__init__.py
    │   └───Data.py
    └───tests
        demo_file.py

And we want to import the Data object into the demo_file in another directory.

1. Go to debug tab in vscode and select "add new launch json file"
2. name "module" variable as your file name in the folder without src code.
3. name "env" variable as the PYTHONPATH directory that holds your src code.
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Module",
            "type": "python",
            "request": "launch",
            "module": "Data",
            // "module": "demo_file",
            "justMyCode": true,
            "env":{
                "PYTHONPATH": "${workspaceFolder}/analyst-d-structur/src"
                // "PYTHONPATH": "${workspaceFolder}/../src"            }
        }
    ]
}
4. hit ctrl+f5 to run
5. setup workspace settings json to allow intelligent tooltips to help with class contructors/
6. ctrl+shift+p
7. select "Open Workspace Settings (JSON)"
8. (not sure exactly what to do here but heres anexample of settings.json):
{
    "python.analysis.extraPaths": ["${workspaceFolder}/analyst-d-structur/src"]
    // "terminal.integrated.env.windows": {
    //     "PYTHONPATH": "${workspaceFolder}/src;${workspaceFolder}/tests"
    // },
    // "python.envFile": "${workspaceFolder}/.env"
}
# Governance chatbot related tools


## How to run firebase emulator

1. cd into this projects root dir, the required firebase.json file is in the root dir of this project:

``` cd /home/sebastian/vsc-workspace/langchain-test ```

2. Change to the nodejs 18 version, here is where the firebase emulator was installed:

```nvm use 18```

3. Run the following command:
   
```firebase emulators:start --import="/home/sebastian/Downloads/polkassembly-prod/2023-05-12T08:11:30_99992"```
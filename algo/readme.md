### OPENAI Key
Set openai key by coping
```
export OPENAI_API_KEY="sk-xxxxxxxxx"
export NEXT_PUBLIC_OPENAI_API_KEY="sk-xxxxxxxxx"
```
to `~/.bashrc`

Or 
```
echo "NEXT_PUBLIC_OPENAI_API_KEY=sk-xxxxxxxxx" >> .env
```
```
echo "OPENAI_API_KEY=sk-xxxxxxxxx" >> .env

```

### Reference
Python  3.8.19

pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1  
cudatoolkit=11.3

transformers    4.39.3
tokenizers  0.15.1

openai  1.28.0
# 👁️ Eye Tracking em tempo real com calibração

Projeto de rastreamento ocular em **tempo real** usando webcam + MediaPipe.

## O que esta versão faz
- Calibração inicial em 4 pontos (cantos da tela).
- Aprende um mapeamento do olhar para coordenadas de tela usando regressão afim.
- Executa tracking contínuo após calibração.
- Envia as coordenadas estimadas para uma API HTTP em JSON.

## Dependências
```bash
pip install opencv-python mediapipe numpy requests
```

## Como executar
1. Ajuste a URL da API no `main.py` em `API_URL` (ex: `http://localhost:8000/gaze`).
2. Rode:
   ```bash
   python main.py
   ```
3. Olhe para cada ponto de calibração até passar para o próximo.
4. Após calibrar, o ponto amarelo representa o local estimado de olhar.
5. Pressione `ESC` para encerrar.

## Payload enviado para API
Exemplo de JSON enviado periodicamente:

```json
{
  "timestamp": 1710000000.12,
  "x": 640.5,
  "y": 300.2,
  "x_norm": 0.500,
  "y_norm": 0.417,
  "frame_width": 1280,
  "frame_height": 720
}
```

## Observações
- A calibração é essencial para qualidade do tracking.
- Ambiente de luz e posição da webcam afetam bastante a precisão.
- Você pode ajustar sensibilidade e taxa de envio no topo do `main.py`.

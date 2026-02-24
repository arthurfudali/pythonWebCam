# 👁️ Eye Tracking em tempo real com calibração

Projeto de rastreamento ocular em **tempo real** usando webcam + MediaPipe, com foco em **desempenho** e **estabilidade** após calibração.

## O que esta versão faz
- Calibração inicial com 5 pontos (4 cantos + centro).
- Coleta robusta por ponto (descarta início da fixação e remove outliers por mediana).
- Mapeamento do olhar para coordenadas de tela via regressão afim.
- Tracking contínuo com suavização temporal.
- Envio assíncrono para API HTTP (não bloqueia o loop principal).
- Processamento em escala reduzida para aumentar FPS.

## Dependências
```bash
pip install opencv-python mediapipe numpy requests
```

## Como executar
1. Ajuste `API_URL` no `main.py` (ex: `http://localhost:8000/gaze`).
2. Rode:
   ```bash
   python main.py
   ```
3. Olhe fixamente para cada ponto de calibração até mudar para o próximo.
4. Após calibrar, o ponto amarelo representa o local estimado de olhar.
5. Pressione `ESC` para encerrar.

## Payload enviado para API
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

## Ajustes importantes
No topo do `main.py` você pode ajustar:
- `PROCESS_SCALE` (ganho de FPS vs detalhe da imagem).
- `CALIBRATION_SECONDS_PER_POINT` (tempo por ponto de calibração).
- `SMOOTHING_ALPHA` (estabilidade vs responsividade).
- `SEND_INTERVAL_SEC` (frequência de envio para API).

# 👁️ Eye Tracking em tempo real (Fullscreen 1920x1080)

Pipeline de rastreamento ocular em tempo real com foco em **estabilidade visual** e calibração robusta.

## O que foi melhorado
- Execução em **fullscreen 1920x1080**.
- Calibração em **15 pontos (grade 5x3)** para maior robustez.
- Mapeamento regularizado para reduzir overfitting.
- Filtro temporal adaptativo + deadzone + snap grid (margem de erro controlada) para reduzir tremor.
- Envio assíncrono para API, mantendo FPS alto.

## Dependências
```bash
pip install opencv-python mediapipe numpy requests
```

## Como executar
```bash
python main.py
```

Durante a calibração:
- mantenha a cabeça o mais estável possível;
- olhe para o ponto vermelho até ele avançar;
- use ambiente bem iluminado.

## Payload enviado
```json
{
  "timestamp": 1710000000.12,
  "x": 962.0,
  "y": 514.0,
  "x_norm": 0.501,
  "y_norm": 0.476,
  "raw_x_norm": 0.493,
  "raw_y_norm": 0.488,
  "frame_width": 1920,
  "frame_height": 1080
}
```

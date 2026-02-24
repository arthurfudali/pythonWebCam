# 👁️ Eye Tracking em tempo real com calibração

Projeto de rastreamento ocular em **tempo real** com foco em FPS e precisão, especialmente no eixo vertical (olhar para cima/baixo).

## Melhorias desta versão
- Calibração em **grade 4x3 (12 pontos)** para aumentar cobertura vertical e horizontal.
- Mapeamento com modelo compacto regularizado (evita overfitting).
- Feature explícita de **vergência** (diferença entre olhos) para ajudar desacoplar olhar de movimento de cabeça.
- Ganho vertical (`VERTICAL_GAIN`) para melhorar sensibilidade de cima/baixo.
- Suavização adaptativa por velocidade + deadzone para reduzir tremor sem perder resposta.
- Payload inclui coordenadas suavizadas e brutas normalizadas.

## Dependências
```bash
pip install opencv-python mediapipe numpy requests
```

## Como executar
1. Ajuste `API_URL` no `main.py`.
2. Rode:
   ```bash
   python main.py
   ```
3. Durante a calibração, mantenha a cabeça parada e mova apenas os olhos.
4. Pressione `ESC` para encerrar.

## Payload enviado para API
```json
{
  "timestamp": 1710000000.12,
  "x": 640.5,
  "y": 300.2,
  "x_norm": 0.500,
  "y_norm": 0.417,
  "raw_x_norm": 0.492,
  "raw_y_norm": 0.431,
  "frame_width": 1280,
  "frame_height": 720
}
```

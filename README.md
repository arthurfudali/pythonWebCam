# 👁️ Eye Tracking em tempo real com calibração

Projeto de rastreamento ocular em **tempo real** com foco em FPS e precisão pós-calibração.

## Melhorias de precisão desta versão
- Calibração em grade **3x3 (9 pontos)** para cobrir melhor toda a tela.
- Features separadas por olho (`lx`, `ly`, `rx`, `ry`) em vez de média simples.
- Mapeamento com **regressão polinomial de 2ª ordem** (com regularização L2), que modela melhor não-linearidades.
- Coleta robusta por ponto com filtro de outliers (mediana/MAD).

## Melhorias de desempenho
- Processamento do Face Mesh em frame reduzido (`PROCESS_SCALE`).
- Envio para API em thread assíncrona (`AsyncGazeSender`) para não bloquear o loop.

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
3. Siga a calibração olhando fixamente para cada ponto.
4. Depois da calibração, o ponto amarelo representa o olhar estimado.
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

export class ConvergenceChart {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private data: number[] = [];
  private padding = { top: 20, right: 20, bottom: 35, left: 55 };
  private accentColor = '#58a6ff';
  private gridColor = '#30363d';
  private textColor = '#8b949e';
  private bgColor = '#0d1117';

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to get 2D context');
    this.ctx = ctx;
    this.resize();
    const ro = new ResizeObserver(() => this.resize());
    ro.observe(canvas);
  }

  private resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
    this.draw();
  }

  update(data: number[]): void {
    this.data = [...data];
    this.draw();
  }

  clear(): void {
    this.data = [];
    this.draw();
  }

  private draw(): void {
    const w = this.canvas.width / (window.devicePixelRatio || 1);
    const h = this.canvas.height / (window.devicePixelRatio || 1);
    const ctx = this.ctx;

    ctx.fillStyle = this.bgColor;
    ctx.fillRect(0, 0, w, h);

    const plotW = w - this.padding.left - this.padding.right;
    const plotH = h - this.padding.top - this.padding.bottom;

    if (plotW <= 0 || plotH <= 0) return;

    ctx.strokeStyle = this.gridColor;
    ctx.lineWidth = 1;

    for (let i = 0; i <= 5; i++) {
      const y = this.padding.top + (plotH * i) / 5;
      ctx.beginPath();
      ctx.moveTo(this.padding.left, y);
      ctx.lineTo(this.padding.left + plotW, y);
      ctx.stroke();
    }

    for (let i = 0; i <= 5; i++) {
      const x = this.padding.left + (plotW * i) / 5;
      ctx.beginPath();
      ctx.moveTo(x, this.padding.top);
      ctx.lineTo(x, this.padding.top + plotH);
      ctx.stroke();
    }

    ctx.fillStyle = this.textColor;
    ctx.font = '11px system-ui';
    ctx.textAlign = 'center';

    ctx.fillText('Iteration', this.padding.left + plotW / 2, h - 8);

    ctx.save();
    ctx.translate(12, this.padding.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('log₁₀(Residual)', 0, 0);
    ctx.restore();

    if (this.data.length < 2) return;

    let minRes = Infinity;
    let maxRes = -Infinity;
    for (const r of this.data) {
      if (r <= 0) continue;
      const logR = Math.log10(r);
      if (logR < minRes) minRes = logR;
      if (logR > maxRes) maxRes = logR;
    }

    if (!isFinite(minRes) || !isFinite(maxRes) || minRes === maxRes) {
      maxRes = 0;
      minRes = -12;
    } else {
      const range = maxRes - minRes;
      maxRes += range * 0.1;
      minRes -= range * 0.1;
    }

    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const y = this.padding.top + (plotH * i) / 5;
      const val = maxRes - ((maxRes - minRes) * i) / 5;
      ctx.fillText(val.toFixed(1), this.padding.left - 6, y + 4);
    }

    ctx.textAlign = 'center';
    const xStep = Math.max(1, Math.floor((this.data.length - 1) / 5));
    for (let i = 0; i < this.data.length; i += xStep) {
      const x = this.padding.left + (plotW * i) / (this.data.length - 1);
      ctx.fillText(String(i), x, this.padding.top + plotH + 16);
    }

    ctx.strokeStyle = this.accentColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < this.data.length; i++) {
      const x = this.padding.left + (plotW * i) / (this.data.length - 1);
      const logR = Math.log10(Math.max(this.data[i], 1e-16));
      const y = this.padding.top + plotH * (1 - (logR - minRes) / (maxRes - minRes));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = this.accentColor;
    for (let i = 0; i < this.data.length; i++) {
      const x = this.padding.left + (plotW * i) / (this.data.length - 1);
      const logR = Math.log10(Math.max(this.data[i], 1e-16));
      const y = this.padding.top + plotH * (1 - (logR - minRes) / (maxRes - minRes));
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

// x402 payment middleware for Hono
import { Hono } from 'hono';

const paymentGuard = async (c, next) => {
  const payment = c.req.header('x-payment');
  if (!payment) {
    return c.json({
      accepts: [{ network: 'eip155:8453', asset: 'USDC', address: Bun.env.X402_WALLET_ADDRESS }],
      price: '0.01',
    }, 402);
  }
  // Verify payment with facilitator
  await next();
};

app.use('/api/*', paymentGuard);

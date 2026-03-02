
"""
QUANTUM TRADER AI - Çekirdek Motor
Kendi kendini geliştiren, adapte olan trading sistemi
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
import logging
import hashlib
import hmac
from collections import deque, defaultdict
import redis.asyncio as redis
import asyncpg
from cryptography.fernet import Ferne
import jwt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    BULL_RANGE = "bull_range"
    BEAR_RANGE = "bear_range"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"


@dataclass
class Trade:
    id: str = field(default_factory=lambda: f"TRADE_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    entry_time: datetime = field(default_factory=datetime.now)
    entry_price: float = 0.0
    side: str = ""  # 'long' or 'short'
    size: float = 0.0
    score_snapshot: Dict[str, float] = field(default_factory=dict)
    regime: str = ""
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl_absolute: float = 0.0
    pnl_percent: float = 0.0
    duration_minutes: float = 0.0
    max_profit: float = 0.0
    max_drawdown: float = 0.0
    fees: float = 0.0
    status: str = "open"
    
    def close(self, exit_price: float, exit_time: datetime, reason: str, fee_rate: float = 0.001):
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.duration_minutes = (exit_time - self.entry_time).total_seconds() / 60
        
        if self.side == 'long':
            self.pnl_percent = (exit_price - self.entry_price) / self.entry_price * 100
        else:
            self.pnl_percent = (self.entry_price - exit_price) / self.entry_price * 100
            
        self.fees = self.size * fee_rate * 2  # Giriş + çıkış
        self.pnl_absolute = (self.size * self.pnl_percent / 100) - self.fees
        self.status = "closed"
        
        return self.pnl_percent


class SecureLogger:
    """Değiştirilemez audit log sistemi"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.chain = []
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        except Exception as e:
            logger.warning(f"Redis bağlantı hatası: {e}")
            self.redis_client = None
        
    def add_block(self, data: dict):
        """Yeni log kaydı ekle (blockchain benzeri)"""
        block = {
            'timestamp': datetime.utcnow().isoformat(),
            'data': data,
            'previous_hash': self.chain[-1]['hash'] if self.chain else '0' * 64,
            'nonce': 0
        }
        
        # Basit proof-of-work
        target = '0000'
        while True:
            block_str = json.dumps(block, sort_keys=True)
            block_hash = hashlib.sha256(block_str.encode()).hexdigest()
            if block_hash.startswith(target):
                break
            block['nonce'] += 1
        
        block['hash'] = block_hash
        self.chain.append(block)
        
        if self.redis_client:
            try:
                asyncio.create_task(self.redis_client.lpush('audit_chain', json.dumps(block)))
            except:
                pass
        
        return block


class MarketAnalyzer:
    """Gelişmiş piyasa analizi"""
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.regime_history = deque(maxlen=50)
        
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Tüm teknik göstergeleri hesapla"""
        if len(data) < 50:
            return self._default_indicators()
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Trend göstergeleri
        ema_9 = close.ewm(span=9).mean().iloc[-1]
        ema_21 = close.ewm(span=21).mean().iloc[-1]
        ema_50 = close.ewm(span=50).mean().iloc[-1]
        
        sma_20 = close.rolling(20).mean().iloc[-1]
        std_20 = close.rolling(20).std().iloc[-1]
        
        # Bollinger Bands
        upper_bb = sma_20 + (std_20 * 2)
        lower_bb = sma_20 - (std_20 * 2)
        bb_width = (upper_bb - lower_bb) / sma_20 if sma_20 > 0 else 0
        bb_position = (close.iloc[-1] - lower_bb) / (upper_bb - lower_bb) if (upper_bb - lower_bb) != 0 else 0.5
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        macd_hist = macd_line.iloc[-1] - signal_line.iloc[-1]
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # ADX (Trend gücü)
        plus_dm = high.diff()
        minus_dm = low.diff()
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=14).mean().iloc[-1]
        adx_value = adx if not pd.isna(adx) else 25
        
        # Volatilite
        returns = close.pct_change().dropna()
        volatility = returns.std() * np.sqrt(365 * 24 * 4) * 100  # Annualized 15min
        
        # Volume Profile
        vol_sma = volume.rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_sma if vol_sma > 0 else 1
        
        # VWAP
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        vwap_distance = (close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1] * 100
        
        # Momentum
        momentum_5 = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        momentum_20 = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0
        
        return {
            'rsi': float(rsi_value),
            'adx': float(adx_value),
            'macd_hist': float(macd_hist),
            'bb_position': float(bb_position),
            'bb_width': float(bb_width),
            'volatility': float(volatility),
            'vol_ratio': float(vol_ratio),
            'vwap_distance': float(vwap_distance),
            'ema_trend': 1 if ema_9 > ema_21 > ema_50 else -1 if ema_9 < ema_21 < ema_50 else 0,
            'momentum_5': float(momentum_5),
            'momentum_20': float(momentum_20),
            'price_vs_sma20': (close.iloc[-1] / sma_20 - 1) * 100
        }
    
    def _default_indicators(self):
        return {
            'rsi': 50, 'adx': 25, 'macd_hist': 0, 'bb_position': 0.5,
            'bb_width': 0.1, 'volatility': 50, 'vol_ratio': 1,
            'vwap_distance': 0, 'ema_trend': 0, 'momentum_5': 0,
            'momentum_20': 0, 'price_vs_sma20': 0
        }
    
    def detect_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Piyasa rejimini tespit et"""
        indicators = self.calculate_all_indicators(data)
        
        vol = indicators['volatility']
        adx = indicators['adx']
        ema_trend = indicators['ema_trend']
        bb_width = indicators['bb_width']
        
        # Volatilite bazlı
        if vol > 100:
            return MarketRegime.HIGH_VOLATILITY
        elif vol < 20:
            return MarketRegime.LOW_VOLATILITY
        
        # Trend bazlı
        if adx > 30:
            if ema_trend == 1:
                return MarketRegime.BULL_TREND
            elif ema_trend == -1:
                return MarketRegime.BEAR_TREND
        
        # Range bazlı
        if ema_trend >= 0:
            return MarketRegime.BULL_RANGE
        else:
            return MarketRegime.BEAR_RANGE


class NeuralStrategy(nn.Module):
    """Derin öğrenme strateji ağı"""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)  # Long, Short, Hold
        )
        
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = attn_out.mean(dim=1)
        
        # Action probabilities
        logits = self.decoder(pooled)
        probs = torch.softmax(logits, dim=-1)
        
        # Confidence score
        conf = self.confidence(pooled)
        
        return probs, conf, attn_weights


class SelfLearningEngine:
    """Kendi kendini geliştiren öğrenme motoru"""
    
    def __init__(self):
        self.trade_memory = deque(maxlen=1000)
        self.models = {}
        self.scaler = StandardScaler()
        self.strategy_params = {
            'entry_threshold': 70,
            'exit_profit': 2.0,
            'exit_loss': -1.0,
            'trailing_stop': True,
            'time_based_exit': 30  # dakika
        }
        self.performance_by_regime = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0})
        
    def extract_features(self, data: pd.DataFrame, indicators: Dict) -> np.ndarray:
        """ML için özellik vektörü"""
        features = []
        
        # Normalize edilmiş göstergeler
        features.extend([
            indicators['rsi'] / 100,
            indicators['adx'] / 100,
            np.tanh(indicators['macd_hist']),
            indicators['bb_position'],
            np.log1p(indicators['volatility']) / 5,
            min(indicators['vol_ratio'] / 3, 2),
            np.tanh(indicators['vwap_distance'] / 10),
            indicators['ema_trend'],
            np.tanh(indicators['momentum_5'] / 10),
            np.tanh(indicators['momentum_20'] / 20),
        ])
        
        # Hacim profili
        vol_changes = data['volume'].pct_change().dropna().tail(5).values
        if len(vol_changes) < 5:
            vol_changes = np.zeros(5)
        features.extend(np.clip(vol_changes / 10, -2, 2))
        
        # Zaman özellikleri
        current_time = data.index[-1]
        features.extend([
            np.sin(2 * np.pi * current_time.hour / 24),
            np.cos(2 * np.pi * current_time.hour / 24),
            current_time.weekday() / 7,
            current_time.month / 12,
        ])
        
        return np.array(features).reshape(1, -1)
    
    def record_trade(self, trade: Trade, features: np.ndarray, regime: str):
        """İşlem kaydet"""
        self.trade_memory.append({
            'trade': trade,
            'features': features.flatten(),
            'regime': regime,
            'timestamp': datetime.now()
        })
        
        # Rejim performansı
        perf = self.performance_by_regime[regime]
        perf['trades'] += 1
        if trade.pnl_percent > 0:
            perf['wins'] += 1
        perf['pnl'] += trade.pnl_percent
    
    def optimize_strategy(self):
        """Strateji parametrelerini optimize et"""
        if len(self.trade_memory) < 50:
            return
        
        recent = list(self.trade_memory)[-100:]
        
        # Basit grid search
        best_pnl = -np.inf
        best_params = self.strategy_params.copy()
        
        for profit in [1.5, 2.0, 2.5, 3.0, 4.0]:
            for loss in [-0.5, -1.0, -1.5, -2.0]:
                total_pnl = 0
                for trade_data in recent:
                    trade = trade_data['trade']
                    # Simülasyon
                    if trade.max_profit >= profit:
                        sim_pnl = profit
                    elif trade.max_drawdown <= loss:
                        sim_pnl = loss
                    else:
                        sim_pnl = trade.pnl_percent
                    total_pnl += sim_pnl
                
                avg_pnl = total_pnl / len(recent)
                if avg_pnl > best_pnl:
                    best_pnl = avg_pnl
                    best_params['exit_profit'] = profit
                    best_params['exit_loss'] = loss
        
        self.strategy_params.update(best_params)
        logger.info(f"Strateji optimize edildi: {best_params}")
    
    def predict_success(self, features: np.ndarray, regime: str) -> float:
        """İşlem başarı olasılığı"""
        # Basit heuristic (gerçekte eğitilmiş model olmalı)
        regime_perf = self.performance_by_regime[regime]
        if regime_perf['trades'] > 10:
            base_prob = regime_perf['wins'] / regime_perf['trades']
        else:
            base_prob = 0.5
        
        # Feature bazlı ayarlama
        rsi = features[0][0] * 100  # 0-100
        if 30 < rsi < 70:
            base_prob += 0.1
        
        return min(max(base_prob, 0.1), 0.9)


class RiskManager:
    """Dinamik risk yönetimi"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.max_drawdown = 0
        self.daily_pnl = 0
        self.last_reset = datetime.now()
        self.daily_limit = initial_capital * 0.03  # %3 günlük limit
        
    def update(self, pnl: float):
        self.current_capital += pnl
        self.daily_pnl += pnl
        
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        dd = (self.peak_capital - self.current_capital) / self.peak_capital
        self.max_drawdown = max(self.max_drawdown, dd)
        
        # Günlük reset
        if (datetime.now() - self.last_reset).days >= 1:
            self.daily_pnl = 0
            self.last_reset = datetime.now()
    
    def can_trade(self) -> bool:
        """Risk limitleri içinde mi?"""
        if self.daily_pnl <= -self.daily_limit:
            logger.warning("Günlük kayıp limitine ulaşıldı!")
            return False
        if self.max_drawdown > 0.10:  # %10 max drawdown
            logger.warning("Max drawdown limiti aşıldı!")
            return False
        return True
    
    def position_size(self, confidence: float, volatility: float) -> float:
        """Kelly criterion benzeri pozisyon boyutu"""
        # Volatilite ayarı
        vol_factor = 1 / (1 + volatility / 50)
        
        # Kelly basitleştirilmiş
        win_rate = 0.55
        avg_win = 2.0
        avg_loss = 1.0
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly = max(0, min(kelly, 0.05))  # Max %5
        
        size = self.current_capital * kelly * vol_factor * confidence
        return min(size, self.current_capital * 0.02)  # Max %2


class UltraAdaptiveBot:
    """Ana bot sınıfı"""
    
    def __init__(self, user_id: str, mode: str = 'paper'):
        self.user_id = user_id
        self.mode = mode  # 'paper', 'live', 'backtest'
        
        self.analyzer = MarketAnalyzer()
        self.learner = SelfLearningEngine()
        self.risk = RiskManager()
        self.secure_log = SecureLogger()
        
        self.active_trade: Optional[Trade] = None
        self.trade_history: List[Trade] = []
        self.is_running = False
        
        # Neural network (basit versiyon)
        self.neural_net = NeuralStrategy()
        self.neural_net.eval()
        
        # Veri buffer
        self.data_buffer = deque(maxlen=200)
        
        logger.info(f"Bot başlatıldı: {user_id} | Mod: {mode}")
    
    async def process_data(self, new_data: pd.DataFrame):
        """Yeni veri işle"""
        self.data_buffer.append(new_data)
        
        if len(self.data_buffer) < 50:
            return
        
        # Veriyi birleştir
        data = pd.concat(list(self.data_buffer)).drop_duplicates()
        
        # Analiz
        regime = self.analyzer.detect_regime(data)
        indicators = self.analyzer.calculate_all_indicators(data)
        features = self.learner.extract_features(data, indicators)
        
        # Tahminler
        success_prob = self.learner.predict_success(features, regime.value)
        params = self.learner.strategy_params
        
        # Karar
        await self.make_decision(data, regime, indicators, features, success_prob, params)
    
    async def make_decision(self, data: pd.DataFrame, regime, indicators, 
                          features, prob, params):
        """İşlem kararı"""
        current_price = data['close'].iloc[-1]
        current_time = data.index[-1]
        
        # Giriş kontrolü
        if self.active_trade is None:
            if not self.risk.can_trade():
                return
            
            entry_score = self.calculate_score(indicators, regime, prob)
            
            if entry_score >= params['entry_threshold'] and prob > 0.6:
                size = self.risk.position_size(prob, indicators['volatility'])
                
                # Rejim bazlı yön
                if regime in [MarketRegime.BULL_TREND, MarketRegime.BULL_RANGE]:
                    side = 'long'
                elif regime in [MarketRegime.BEAR_TREND, MarketRegime.BEAR_RANGE]:
                    side = 'short'
                else:
                    side = 'long' if indicators['ema_trend'] >= 0 else 'short'
                
                self.active_trade = Trade(
                    entry_time=current_time,
                    entry_price=current_price,
                    side=side,
                    size=size,
                    score_snapshot={
                        'entry_score': entry_score,
                        'success_prob': prob,
                        'indicators': indicators
                    },
                    regime=regime.value
                )
                
                logger.info(f"🟢 {side.upper()} AÇILDI @ {current_price:.2f} | "
                          f"Skor: {entry_score:.1f} | Prob: {prob:.2f}")
                
                self.secure_log.add_block({
                    'event': 'trade_opened',
                    'trade_id': self.active_trade.id,
                    'side': side,
                    'price': current_price,
                    'size': size
                })
        
        else:
            # Çıkış kontrolü
            await self.check_exit(data, current_price, current_time, params)
    
    def calculate_score(self, indicators: Dict, regime, prob: float) -> float:
        """Giriş skoru hesapla"""
        score = 50
        
        # RSI
        if 30 < indicators['rsi'] < 70:
            score += 15
        elif indicators['rsi'] < 30:
            score += 25
        
        # Trend
        if indicators['adx'] > 25:
            score += 15
        
        # Volatilite
        if 20 < indicators['volatility'] < 80:
            score += 10
        
        # Hacim
        if indicators['vol_ratio'] > 1.2:
            score += 10
        
        # ML güveni
        score += prob * 20
        
        # Rejim bonusu
        if regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
            score += 10
        
        return min(100, score)
    
    async def check_exit(self, data, current_price, current_time, params):
        """Çıkış kontrolü"""
        trade = self.active_trade
        
        # PnL hesapla
        if trade.side == 'long':
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price * 100
        else:
            pnl_pct = (trade.entry_price - current_price) / trade.entry_price * 100
        
        trade.max_profit = max(trade.max_profit, pnl_pct)
        trade.max_drawdown = min(trade.max_drawdown, pnl_pct)
        
        hold_time = (current_time - trade.entry_time).total_seconds() / 60
        
        exit_reason = None
        
        # Take profit
        if pnl_pct >= params['exit_profit']:
            exit_reason = f"take_profit_{pnl_pct:.1f}%"
        
        # Stop loss
        elif pnl_pct <= params['exit_loss']:
            exit_reason = f"stop_loss_{pnl_pct:.1f}%"
        
        # Trailing stop
        elif trade.max_profit > params['exit_profit'] * 0.6 and pnl_pct < trade.max_profit - 1.0:
            exit_reason = "trailing_stop"
        
        # Zaman limiti
        elif hold_time >= params['time_based_exit']:
            exit_reason = "time_limit"
        
        # Rejim değişimi
        elif hold_time > 15:
            new_regime = self.analyzer.detect_regime(data)
            if ((trade.side == 'long' and new_regime in [MarketRegime.BEAR_TREND, MarketRegime.BEAR_RANGE]) or
                (trade.side == 'short' and new_regime in [MarketRegime.BULL_TREND, MarketRegime.BULL_RANGE])):
                exit_reason = "regime_change"
        
        if exit_reason:
            await self.close_trade(current_price, current_time, exit_reason, pnl_pct)
    
    async def close_trade(self, exit_price, exit_time, reason, pnl_pct):
        """İşlemi kapat"""
        trade = self.active_trade
        
        # Özellikleri kaydet
        data = pd.concat(list(self.data_buffer))
        indicators = self.analyzer.calculate_all_indicators(data)
        features = self.learner.extract_features(data, indicators)
        
        # Kapat
        actual_pnl = trade.close(exit_price, exit_time, reason)
        self.trade_history.append(trade)
        
        # Risk ve öğrenme güncelle
        self.risk.update(trade.pnl_absolute)
        self.learner.record_trade(trade, features, trade.regime)
        
        # Log
        emoji = "✅" if actual_pnl > 0 else "❌"
        logger.info(f"{emoji} KAPANDI ({reason}) | PnL: {actual_pnl:.2f}% | "
                   f"Süre: {trade.duration_minutes:.0f}dk | "
                   f"Sermaye: {self.risk.current_capital:.2f}")
        
        self.secure_log.add_block({
            'event': 'trade_closed',
            'trade_id': trade.id,
            'reason': reason,
            'pnl': actual_pnl
        })
        
        self.active_trade = None
        
        # Her 20 işlemde optimize et
        if len(self.trade_history) % 20 == 0:
            self.learner.optimize_strategy()
    
    async def backtest(self, historical_data: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """Backtest çalıştır"""
        self.risk = RiskManager(initial_capital)
        self.trade_history = []
        
        results = {
            'trades': [],
            'equity_curve': [],
            'metrics': {}
        }
        
        for i in range(50, len(historical_data)):
            window = historical_data.iloc[:i]
            await self.process_data(window)
            
            if i % 100 == 0:
                results['equity_curve'].append({
                    'time': window.index[-1],
                    'equity': self.risk.current_capital
                })
        
        # Metrikler
        if self.trade_history:
            pnls = [t.pnl_percent for t in self.trade_history]
            wins = len([p for p in pnls if p > 0])
            
            results['metrics'] = {
                'total_trades': len(self.trade_history),
                'winning_trades': wins,
                'win_rate': wins / len(pnls) * 100,
                'total_return': (self.risk.current_capital - initial_capital) / initial_capital * 100,
                'max_drawdown': self.risk.max_drawdown * 100,
                'sharpe_ratio': np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0,
                'profit_factor': abs(sum([p for p in pnls if p > 0])) / abs(sum([p for p in pnls if p < 0])) if sum([p for p in pnls if p < 0]) != 0 else float('inf')
            }
            
            results['trades'] = [
                {
                    'id': t.id,
                    'side': t.side,
                    'entry': t.entry_price,
                    'exit': t.exit_price,
                    'pnl': t.pnl_percent,
                    'duration': t.duration_minutes,
                    'regime': t.regime
                }
                for t in self.trade_history
            ]
        
        return results


# Test ve çalıştırma
if __name__ == "__main__":
    # Test verisi oluştur
    np.random.seed(42)
    n = 2000
    dates = pd.date_range('2024-01-01', periods=n, freq='15min')
    
    # Trend + gürültü
    trend = np.sin(np.linspace(0, 4*np.pi, n)) * 0.05
    noise = np.random.normal(0, 0.01, n)
    returns = trend + noise
    
    # Volatilite patlamaları
    returns[400:500] *= 3
    returns[1200:1300] *= 2
    
    prices = 50000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n)),
        'high': prices * (1 + abs(np.random.normal(0, 0.002, n))),
        'low': prices * (1 - abs(np.random.normal(0, 0.002, n))),
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, n)
    }, index=dates)
    
    # Bot oluştur ve backtest yap
    async def main():
        bot = UltraAdaptiveBot("test_user", "backtest")
        results = await bot.backtest(df, initial_capital=10000)
        
        print("\n" + "="*60)
        print("BACKTEST SONUÇLARI")
        print("="*60)
        print(f"Toplam İşlem: {results['metrics']['total_trades']}")
        print(f"Kazanma Oranı: {results['metrics']['win_rate']:.1f}%")
        print(f"Toplam Getiri: {results['metrics']['total_return']:.2f}%")
        print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
        print(f"Profit Factor: {results['metrics']['profit_factor']:.2f}")
        print("="*60)
    
    asyncio.run(main())
EOF

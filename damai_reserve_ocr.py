#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大麦网 抢票预约自动化 — OCR 版

进入演出详情页后，WebView 和 Native 方式均无法获取页面信息，
因此使用 OCR（截屏 + 文字识别 + 坐标点击）实现：
  1. 点击底部「已预约」按钮进入预约详情页
  2. 在预约详情页提取场次和票档信息

OCR 引擎（自动检测，优先级递减）：
  1. PaddleOCR  — 中文识别最好（pip install paddleocr paddlepaddle）
  2. RapidOCR   — 轻量级 ONNX（pip install rapidocr_onnxruntime）
  3. pytesseract — 通用兜底（pip install pytesseract + 系统 Tesseract-OCR）

使用：
  python damai_reserve_ocr.py                                    # 交互式
  python damai_reserve_ocr.py --phone 1********5               # 指定手机号
  python damai_reserve_ocr.py --device abc123                   # 指定设备序列号
  python damai_reserve_ocr.py --skip-login                      # 跳过登录
  python damai_reserve_ocr.py --verbose                          # 详细日志

依赖：
  pip install uiautomator2 requests websocket-client
  pip install paddleocr paddlepaddle          # 推荐
  # 或
  pip install rapidocr_onnxruntime            # 轻量替代
  Android 设备需开启 USB 调试并通过 adb 连接
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from damai_reserve_u2 import (
    DamaiReserveAutomation,
    try_load_cookies,
    DEFAULT_PHONE,
    DEFAULT_COOKIE_FILE,
)
from damai_u2 import DAMAI_PACKAGE


# ─── OCR 引擎抽象 ────────────────────────────────────────────────────────

# OCR 引擎类型常量
OCR_PADDLE = "paddleocr"
OCR_RAPID = "rapidocr"
OCR_TESSERACT = "tesseract"
OCR_NONE = "none"


def _detect_ocr_engine() -> str:
    """检测可用的 OCR 引擎，返回引擎类型常量。"""
    # 1. PaddleOCR
    try:
        from paddleocr import PaddleOCR  # noqa: F401
        return OCR_PADDLE
    except ImportError:
        pass

    # 2. RapidOCR
    try:
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401
        return OCR_RAPID
    except ImportError:
        pass

    # 3. pytesseract
    try:
        import pytesseract  # noqa: F401
        return OCR_TESSERACT
    except ImportError:
        pass

    return OCR_NONE


# ─── OCR 版自动化类 ──────────────────────────────────────────────────────

class DamaiReserveOCRAutomation(DamaiReserveAutomation):
    """大麦网抢票预约自动化（OCR 版）。

    继承 DamaiReserveAutomation，重写演出详情页相关方法：
      - click_reserved_button()：OCR 截屏识别「已预约」并点击
      - extract_sessions_and_tickets()：OCR 截屏识别场次和票档
    """

    def __init__(
        self,
        device_serial: Optional[str] = None,
        verbose: bool = False,
        ocr_engine: Optional[str] = None,
    ):
        """
        Args:
            device_serial: 设备序列号，None 则自动检测
            verbose: 是否输出详细日志
            ocr_engine: 指定 OCR 引擎（paddleocr/rapidocr/tesseract/auto），
                        默认 auto 自动检测
        """
        super().__init__(device_serial=device_serial, verbose=verbose)

        # OCR 引擎
        self._ocr_engine = None      # OCR 实例
        self._ocr_type = OCR_NONE    # 引擎类型

        if ocr_engine and ocr_engine != "auto":
            self._ocr_type = ocr_engine
        else:
            self._ocr_type = _detect_ocr_engine()

        # 截屏临时目录
        self._screenshot_dir = tempfile.mkdtemp(prefix="damai_ocr_")

    # ── OCR 初始化 ──────────────────────────────────────────────────────

    def _init_ocr(self) -> bool:
        """初始化 OCR 引擎。

        Returns:
            是否成功初始化
        """
        if self._ocr_engine is not None:
            return True

        engine_type = self._ocr_type

        if engine_type == OCR_PADDLE:
            try:
                from paddleocr import PaddleOCR
                print("🔧 正在初始化 PaddleOCR（首次运行会下载模型，请耐心等待）…")
                self._ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang="ch",
                    show_log=self.verbose,
                    use_gpu=False,
                )
                print("✅ PaddleOCR 初始化成功")
                return True
            except Exception as e:
                print(f"⚠️ PaddleOCR 初始化失败：{e}")
                # 尝试回退
                self._ocr_type = _detect_ocr_engine()
                if self._ocr_type != OCR_NONE and self._ocr_type != OCR_PADDLE:
                    print(f"  ↳ 回退到 {self._ocr_type}")
                    return self._init_ocr()
                return False

        elif engine_type == OCR_RAPID:
            try:
                from rapidocr_onnxruntime import RapidOCR
                print("🔧 正在初始化 RapidOCR…")
                self._ocr_engine = RapidOCR()
                print("✅ RapidOCR 初始化成功")
                return True
            except Exception as e:
                print(f"⚠️ RapidOCR 初始化失败：{e}")
                # 尝试回退到 tesseract
                self._ocr_type = OCR_TESSERACT
                return self._init_ocr()

        elif engine_type == OCR_TESSERACT:
            try:
                import pytesseract
                from PIL import Image  # noqa: F401
                print("🔧 正在初始化 pytesseract…")
                # pytesseract 不需要实例化，直接使用模块函数
                self._ocr_engine = pytesseract
                print("✅ pytesseract 初始化成功")
                return True
            except Exception as e:
                print(f"⚠️ pytesseract 初始化失败：{e}")
                return False

        else:
            print("❌ 未找到可用的 OCR 引擎！")
            print("请安装其中一个：")
            print("  pip install paddleocr paddlepaddle          # 推荐")
            print("  pip install rapidocr_onnxruntime            # 轻量替代")
            print("  pip install pytesseract                     # 通用兜底")
            return False

    # ── 截屏 ────────────────────────────────────────────────────────────

    def _take_screenshot(self) -> str:
        """通过 uiautomator2 截屏。

        Returns:
            截图文件路径
        """
        if not self.d:
            raise RuntimeError("设备未连接")

        ts = int(time.time() * 1000)
        path = os.path.join(self._screenshot_dir, f"screenshot_{ts}.png")
        self.d.screenshot(path)
        self._log(f"截屏保存到：{path}")
        return path

    # ── OCR 识别 ────────────────────────────────────────────────────────

    def _ocr_recognize(self, image_path: str) -> List[Dict[str, Any]]:
        """对截图执行 OCR 识别。

        Args:
            image_path: 截图文件路径

        Returns:
            识别结果列表，每项为 dict：
              text: 识别的文字
              bbox: 边界框 [x1, y1, x2, y2]
              confidence: 置信度 (0~1)
        """
        if not self._init_ocr():
            return []

        results: List[Dict[str, Any]] = []

        if self._ocr_type == OCR_PADDLE:
            results = self._ocr_recognize_paddle(image_path)
        elif self._ocr_type == OCR_RAPID:
            results = self._ocr_recognize_rapid(image_path)
        elif self._ocr_type == OCR_TESSERACT:
            results = self._ocr_recognize_tesseract(image_path)

        self._log(f"OCR 识别到 {len(results)} 个文字区域")
        return results

    def _ocr_recognize_paddle(self, image_path: str) -> List[Dict[str, Any]]:
        """PaddleOCR 识别。"""
        results = []
        try:
            ocr_result = self._ocr_engine.ocr(image_path, cls=True)
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    # line: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)]
                    bbox_points = line[0]
                    text, confidence = line[1]
                    # 转换为 [x1, y1, x2, y2] 格式
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    results.append({
                        "text": text,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                    })
        except Exception as e:
            self._log(f"PaddleOCR 识别失败：{e}")
        return results

    def _ocr_recognize_rapid(self, image_path: str) -> List[Dict[str, Any]]:
        """RapidOCR 识别。"""
        results = []
        try:
            from PIL import Image
            img = Image.open(image_path)
            ocr_result, _ = self._ocr_engine(img)
            if ocr_result:
                for item in ocr_result:
                    # item: [bbox, text, confidence]
                    bbox_points = item[0]
                    text = item[1]
                    confidence = item[2]
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    results.append({
                        "text": text,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                    })
        except Exception as e:
            self._log(f"RapidOCR 识别失败：{e}")
        return results

    def _ocr_recognize_tesseract(self, image_path: str) -> List[Dict[str, Any]]:
        """pytesseract 识别。"""
        results = []
        try:
            from PIL import Image
            img = Image.open(image_path)
            # 使用中文+英文语言包
            data = self._ocr_engine.image_to_data(
                img, lang="chi_sim+eng", output_type=self._ocr_engine.Output.DICT
            )
            n_boxes = len(data["text"])
            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])
                if not text or conf < 30:
                    continue
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                results.append({
                    "text": text,
                    "bbox": [x, y, x + w, y + h],
                    "confidence": conf / 100.0,
                })
        except Exception as e:
            self._log(f"pytesseract 识别失败：{e}")
        return results

    # ── OCR 查找并点击 ──────────────────────────────────────────────────

    def _ocr_find_and_click(
        self,
        keyword: str,
        region: str = "full",
        scroll_retry: int = 0,
        scroll_direction: str = "down",
        confidence_threshold: float = 0.5,
    ) -> bool:
        """OCR 查找含 keyword 的文字区域并点击其中心坐标。

        Args:
            keyword: 要查找的关键词
            region: 搜索区域，"full" 全屏 / "bottom" 下半屏 / "top" 上半屏
            scroll_retry: 滚动重试次数（0 = 不滚动重试）
            scroll_direction: 滚动方向 "down" 向下 / "up" 向上
            confidence_threshold: 置信度阈值

        Returns:
            是否成功找到并点击
        """
        if not self.d:
            return False

        # 获取屏幕尺寸
        window_size = self.d.window_size()
        screen_w = window_size[0]
        screen_h = window_size[1]

        # 计算搜索区域的 y 范围
        if region == "bottom":
            y_min, y_max = screen_h * 0.5, screen_h
        elif region == "top":
            y_min, y_max = 0, screen_h * 0.5
        else:
            y_min, y_max = 0, screen_h

        for attempt in range(scroll_retry + 1):
            if attempt > 0:
                # 滚动页面
                if scroll_direction == "down":
                    self.d.swipe(0.5, 0.8, 0.5, 0.2)
                else:
                    self.d.swipe(0.5, 0.2, 0.5, 0.8)
                time.sleep(1)
                self._log(f"滚动重试第 {attempt} 次")

            # 截屏 + OCR
            screenshot_path = self._take_screenshot()
            ocr_results = self._ocr_recognize(screenshot_path)

            # 查找含 keyword 的文字
            matches = []
            for item in ocr_results:
                text = item.get("text", "")
                conf = item.get("confidence", 0)
                bbox = item.get("bbox", [])
                if len(bbox) != 4:
                    continue

                # 关键词匹配
                if keyword not in text:
                    continue

                # 置信度过滤
                if conf < confidence_threshold:
                    continue

                # 区域过滤
                _, y1, _, y2 = bbox
                center_y = (y1 + y2) / 2
                if center_y < y_min or center_y > y_max:
                    continue

                matches.append(item)

            if matches:
                # 选择置信度最高的匹配
                best = max(matches, key=lambda m: m.get("confidence", 0))
                bbox = best["bbox"]
                # 计算中心坐标
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                self._log(
                    f"OCR 找到「{keyword}」：text={best['text']!r}, "
                    f"bbox={bbox}, center=({cx:.0f}, {cy:.0f}), "
                    f"conf={best['confidence']:.2f}"
                )

                # 点击
                self.d.click(cx, cy)
                self._log(f"已点击坐标 ({cx:.0f}, {cy:.0f})")
                return True

        return False

    # ── OCR 提取文字 ────────────────────────────────────────────────────

    def _ocr_extract_all_text(
        self,
        max_scrolls: int = 5,
        confidence_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """OCR 提取当前页面所有文字（支持滚动翻页）。

        Args:
            max_scrolls: 最大滚动次数
            confidence_threshold: 置信度阈值

        Returns:
            所有识别结果列表（去重后）
        """
        all_results: List[Dict[str, Any]] = []
        seen_texts: set = set()

        for scroll_idx in range(max_scrolls + 1):
            if scroll_idx > 0:
                # 向上滑动（查看下方内容）
                self.d.swipe(0.5, 0.7, 0.5, 0.3)
                time.sleep(1.5)  # 等待页面稳定

            screenshot_path = self._take_screenshot()
            ocr_results = self._ocr_recognize(screenshot_path)

            new_count = 0
            for item in ocr_results:
                text = item.get("text", "").strip()
                conf = item.get("confidence", 0)
                if not text or conf < confidence_threshold:
                    continue
                # 去重（基于文字内容）
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                all_results.append(item)
                new_count += 1

            self._log(f"第 {scroll_idx} 次截屏：识别 {len(ocr_results)} 项，新增 {new_count} 项")

            # 如果连续两次没有新内容，停止滚动
            if scroll_idx > 0 and new_count == 0:
                self._log("无新内容，停止滚动")
                break

        return all_results

    # ── 重写：获取第一条已预约演出 ──────────────────────────────────────

    # 常见非演出文字（页面标题、导航栏、Tab 标签等）
    _SKIP_TEXTS = {
        "抢票预约", "我的", "搜索", "首页", "更多", "返回", "设置",
        "我的预约", "预约记录", "全部", "进行中", "已结束",
        "编辑", "删除", "取消", "确定", "关闭",
        "大麦", "damai", "推荐", "关注",
    }

    def get_first_reserved_show(self) -> Optional[Dict[str, str]]:
        """获取第一条已预约的演出（修正版）。

        当前页面是「我的抢票预约」列表页，Native/WebView 均可用，
        但原实现选择器太宽泛会匹配到页面标题等非演出元素。

        修正策略：
          1. 等待列表加载完成
          2. 获取屏幕尺寸，跳过顶部标题栏区域（约前 15% 高度）
          3. 查找所有 TextView，按 y 坐标排序
          4. 跳过已知非演出文字和过短文字
          5. 取内容区域中第一个看起来像演出名的文字并点击

        Returns:
            dict with keys: name, url, item_id; or None
        """
        print("🎭 正在查找第一条已预约演出…")

        result: Dict[str, str] = {}

        # ── Native 层（精准定位）──
        try:
            self.switch_to_native()
            time.sleep(2)  # 等待列表加载

            # 获取屏幕尺寸，用于跳过顶部标题栏
            window_size = self.d.window_size()
            screen_h = window_size[1]
            # 标题栏大约占屏幕顶部 12%，跳过该区域
            title_bar_bottom = screen_h * 0.12

            # 查找所有 TextView
            text_views = self.d(className="android.widget.TextView")
            if not text_views.exists(timeout=5):
                self._log("页面上没有 TextView")
            else:
                count = text_views.count
                self._log(f"页面上共 {count} 个 TextView")

                # 收集候选演出名：[(y中心坐标, index, text), ...]
                candidates: List[Tuple[float, int, str]] = []

                for idx in range(min(count, 50)):
                    try:
                        tv = text_views[idx]
                        txt = (tv.get_text() or "").strip()
                        if not txt:
                            continue

                        # 跳过过短文字（演出名通常 > 4 字符）
                        if len(txt) <= 4:
                            continue

                        # 跳过已知非演出文字
                        if txt in self._SKIP_TEXTS:
                            self._log(f"跳过非演出文字：{txt!r}")
                            continue

                        # 跳过包含常见 UI 关键词的短文字
                        ui_keywords = ["预约", "登录", "注册", "搜索", "首页", "我的"]
                        if any(kw in txt for kw in ui_keywords) and len(txt) <= 8:
                            self._log(f"跳过 UI 文字：{txt!r}")
                            continue

                        # 获取元素位置，跳过标题栏区域
                        try:
                            bounds = tv.bounds()
                            # bounds 返回 (left, top, right, bottom)
                            if len(bounds) == 4:
                                center_y = (bounds[1] + bounds[3]) / 2
                            else:
                                center_y = 0
                        except Exception:
                            center_y = 0

                        if center_y > 0 and center_y < title_bar_bottom:
                            self._log(f"跳过标题栏区域文字：{txt!r} (y={center_y:.0f})")
                            continue

                        candidates.append((center_y, idx, txt))
                        self._log(f"候选演出：{txt!r} (y={center_y:.0f}, idx={idx})")

                    except Exception as e:
                        self._log(f"遍历 TextView[{idx}] 失败：{e}")
                        continue

                # 按 y 坐标排序，取最靠上的（即列表中第一条）
                if candidates:
                    candidates.sort(key=lambda c: c[0])
                    _, best_idx, best_text = candidates[0]
                    result["name"] = best_text
                    self._log(f"选择第一条演出：{best_text!r} (idx={best_idx})")

                    # 点击进入演出详情
                    text_views[best_idx].click()
                    time.sleep(3)
                    print(f"✅ 已点击第一条预约演出：{best_text}")
                    return result

        except Exception as e:
            self._log(f"Native 精准定位失败：{e}")

        # ── 备用：尝试通过 RecyclerView/ListView 的子项定位 ──
        try:
            self.switch_to_native()
            time.sleep(1)

            # 查找 RecyclerView 或 ListView
            for list_class in [
                "androidx.recyclerview.widget.RecyclerView",
                "android.widget.ListView",
                "android.support.v7.widget.RecyclerView",
            ]:
                list_view = self.d(className=list_class)
                if list_view.exists(timeout=3):
                    self._log(f"找到列表容器：{list_class}")
                    # 获取第一个子项
                    first_child = list_view.child(index=0)
                    if first_child.exists(timeout=2):
                        # 尝试获取子项中的文字
                        child_text = first_child.child(
                            className="android.widget.TextView"
                        )
                        if child_text.exists(timeout=2):
                            txt = (child_text.get_text() or "").strip()
                            if txt and len(txt) > 4:
                                result["name"] = txt
                                self._log(f"列表第一项文字：{txt!r}")

                        first_child.click()
                        time.sleep(3)
                        print(f"✅ 已点击列表第一条演出")
                        return result

        except Exception as e:
            self._log(f"列表容器定位失败：{e}")

        # ── 备用：WebView 层 ──
        self._ensure_webview_connected()
        if self._wd:
            try:
                raw = self._wd.execute_script(
                    """
                    (function() {
                        // 查找演出卡片/链接（优先匹配预约列表中的项）
                        var selectors = [
                            '.reserve-item a', '.booking-item a',
                            '.reserve-list .item a', '.booking-list .item a',
                            '[class*="reserve"] a', '[class*="booking"] a',
                            '[class*="item"] a', '[class*="card"] a',
                            '.list-item a', 'a[href*="item"]', 'a[href*="detail"]'
                        ];
                        for (var s = 0; s < selectors.length; s++) {
                            var el = document.querySelector(selectors[s]);
                            if (el) {
                                return JSON.stringify({
                                    href: el.href || '',
                                    text: (el.textContent || '').trim().substring(0, 100)
                                });
                            }
                        }
                        // 备用：找所有 a 标签中含演出链接的
                        var links = document.querySelectorAll('a[href]');
                        for (var i = 0; i < links.length; i++) {
                            var href = links[i].href || '';
                            if (/item\\.htm|\\/item\\/|itemId=|\\/detail\\//.test(href)) {
                                return JSON.stringify({
                                    href: href,
                                    text: (links[i].textContent || '').trim().substring(0, 100)
                                });
                            }
                        }
                        return null;
                    })();
                    """
                )
                if raw:
                    info = json.loads(raw)
                    result["name"] = info.get("text", "")
                    result["url"] = info.get("href", "")
                    if result.get("url"):
                        item_id = self._extract_item_id_from_url(result["url"])
                        if item_id:
                            result["item_id"] = item_id
                    self._log(f"WebView 找到第一条预约演出：{result}")

                    # 点击进入详情
                    if result.get("url"):
                        self.navigate_to_url(result["url"])
                        time.sleep(3)
                    else:
                        self._wd.execute_script(
                            """
                            var el = document.querySelector('.reserve-item a, .booking-item a, [class*="item"] a, [class*="card"] a');
                            if (el) { el.click(); }
                            """
                        )
                        time.sleep(3)

                    print(f"✅ 已通过 WebView 点击第一条预约演出")
                    return result

            except Exception as e:
                self._log(f"WebView 获取预约演出失败：{e}")

        print("❌ 未找到已预约的演出")
        return None

    # ── 重写：点击「已预约」按钮 ────────────────────────────────────────

    def click_reserved_button(self) -> bool:
        """在演出详情页点击底部「已预约」按钮（OCR 版）。

        策略：
          1. 截屏 + OCR 识别所有文字
          2. 在屏幕下半部分查找「已预约」
          3. 找到 → 点击其中心坐标
          4. 未找到 → 滚动到底部重试（最多 5 次）
          5. 备用：查找「预约」关键词

        Returns:
            是否成功点击
        """
        print("📌 正在通过 OCR 查找「已预约」按钮…")

        # 确保 OCR 已初始化
        if not self._init_ocr():
            print("❌ OCR 引擎初始化失败，无法继续")
            return False

        try:
            self.switch_to_native()
            time.sleep(1)

            # 策略 1：在下半屏查找「已预约」
            if self._ocr_find_and_click(
                keyword="已预约",
                region="bottom",
                scroll_retry=5,
                scroll_direction="down",
            ):
                print("✅ 已通过 OCR 点击「已预约」")
                time.sleep(2)
                return True

            # 策略 2：全屏查找「已预约」
            if self._ocr_find_and_click(
                keyword="已预约",
                region="full",
                scroll_retry=3,
                scroll_direction="down",
            ):
                print("✅ 已通过 OCR 点击「已预约」（全屏搜索）")
                time.sleep(2)
                return True

            # 策略 3：查找「预约」（更宽泛）
            if self._ocr_find_and_click(
                keyword="预约",
                region="bottom",
                scroll_retry=3,
                scroll_direction="down",
            ):
                print("✅ 已通过 OCR 点击含「预约」的按钮")
                time.sleep(2)
                return True

            # 策略 4：调试输出 — 打印当前屏幕所有 OCR 文字
            print("⚠️ 未找到「已预约」按钮，输出当前屏幕 OCR 结果供调试：")
            screenshot_path = self._take_screenshot()
            ocr_results = self._ocr_recognize(screenshot_path)
            for item in ocr_results:
                text = item.get("text", "")
                conf = item.get("confidence", 0)
                bbox = item.get("bbox", [])
                if text and conf > 0.3:
                    print(f"  📝 {text!r}  conf={conf:.2f}  bbox={bbox}")

        except Exception as e:
            self._warn(f"OCR 查找「已预约」按钮异常：{e}")

        print("❌ 未找到「已预约」按钮")
        return False

    # ── 重写：提取场次和票档信息 ────────────────────────────────────────

    def extract_sessions_and_tickets(self) -> Dict[str, Any]:
        """提取场次和票档信息（OCR 版）。

        策略：
          1. 多次截屏 + OCR（支持滚动翻页）
          2. 对所有识别文字按关键词分类：
             - 场次：含日期格式（2025-07-21、7月21日、周X 等）
             - 票档：含价格格式（¥、元、价格、票档 等）
          3. 返回结构化结果

        Returns:
            dict with keys:
              sessions: List[Dict] — 场次列表
              tickets: List[Dict] — 票档列表
              raw_text: str — 页面原始文本（兜底）
        """
        print("📊 正在通过 OCR 提取场次和票档信息…")

        info: Dict[str, Any] = {
            "sessions": [],
            "tickets": [],
            "raw_text": "",
        }

        # 确保 OCR 已初始化
        if not self._init_ocr():
            print("❌ OCR 引擎初始化失败，无法提取信息")
            return info

        try:
            self.switch_to_native()
            time.sleep(1)

            # OCR 提取所有文字（含滚动翻页）
            all_ocr = self._ocr_extract_all_text(max_scrolls=5)

            # 收集所有文字
            all_texts = [item.get("text", "") for item in all_ocr if item.get("text")]
            info["raw_text"] = "\n".join(all_texts)

            # 按关键词分类
            for item in all_ocr:
                text = item.get("text", "").strip()
                if not text:
                    continue

                # ── 场次识别 ──
                # 含日期格式
                if re.search(
                    r'\d{4}[.-]\d{1,2}[.-]\d{1,2}|'
                    r'\d{1,2}月\d{1,2}日|'
                    r'周[一二三四五六日天]|'
                    r'\d{1,2}:\d{2}|'
                    r'\d{1,2}：\d{2}',
                    text
                ):
                    # 排除纯时间（如 14:30）如果没有场次上下文
                    if not re.match(r'^\d{1,2}[:：]\d{2}$', text):
                        info["sessions"].append({"text": text})
                        continue

                # 含场次关键词
                if re.search(r'场次|场次|第[一二三四五六七八九十\d]+场|演出时间', text):
                    info["sessions"].append({"text": text})
                    continue

                # ── 票档识别 ──
                # 含价格格式
                if re.search(r'¥|￥|价格|票档|票价', text):
                    info["tickets"].append({"text": text})
                    continue

                # 含数字+元
                if re.search(r'\d+元|\d+\.\d+元', text):
                    info["tickets"].append({"text": text})
                    continue

                # 含票档关键词
                if re.search(r'票档|座位|席别|看台|内场|VIP|普通|特惠', text):
                    info["tickets"].append({"text": text})
                    continue

            self._log(
                f"OCR 提取到 {len(info['sessions'])} 个场次, "
                f"{len(info['tickets'])} 个票档, "
                f"共 {len(all_texts)} 个文字"
            )

            # 如果结构化信息为空，尝试更宽松的匹配
            if not info["sessions"] and not info["tickets"]:
                self._log("结构化信息为空，尝试宽松匹配…")
                for item in all_ocr:
                    text = item.get("text", "").strip()
                    if not text or len(text) < 2:
                        continue

                    # 宽松场次：含数字+月/日/周
                    if re.search(r'\d+月|\d+日|周\d|场次', text):
                        if not any(s["text"] == text for s in info["sessions"]):
                            info["sessions"].append({"text": text})

                    # 宽松票档：含 ¥/元/票/价/座
                    if re.search(r'¥|元|票|价|座|档|VIP|看台|内场', text):
                        if not any(t["text"] == text for t in info["tickets"]):
                            info["tickets"].append({"text": text})

        except Exception as e:
            self._warn(f"OCR 提取场次票档异常：{e}")

        return info

    # ── 清理 ────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """清理资源（含临时截图目录）。"""
        super().cleanup()

        # 删除临时截图目录
        try:
            import shutil
            if os.path.isdir(self._screenshot_dir):
                shutil.rmtree(self._screenshot_dir, ignore_errors=True)
                self._log(f"已删除临时目录：{self._screenshot_dir}")
        except Exception:
            pass


# ─── 主流程 ──────────────────────────────────────────────────────────────

def main() -> None:
    # Windows 控制台 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="大麦网 抢票预约自动化 — OCR 版（截屏识别 + 坐标点击）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python damai_reserve_ocr.py\n"
            "  python damai_reserve_ocr.py --phone 15757176315\n"
            "  python damai_reserve_ocr.py --device abc123\n"
            "  python damai_reserve_ocr.py --skip-login\n"
            "  python damai_reserve_ocr.py --verbose\n"
            "  python damai_reserve_ocr.py --ocr rapidocr\n"
            "\n"
            "OCR 引擎:\n"
            "  auto       自动检测（默认）\n"
            "  paddleocr  PaddleOCR（中文最好，依赖较重）\n"
            "  rapidocr   RapidOCR（轻量 ONNX 版）\n"
            "  tesseract  pytesseract（通用兜底）\n"
            "\n"
            "依赖:\n"
            "  pip install uiautomator2 requests websocket-client\n"
            "  pip install paddleocr paddlepaddle          # 推荐\n"
            "  pip install rapidocr_onnxruntime            # 轻量替代\n"
            "\n"
            "设备准备:\n"
            "  1. 手机开启 USB 调试（设置 → 开发者选项 → USB 调试）\n"
            "  2. USB 连接电脑，运行 adb devices 确认设备可见\n"
        ),
    )
    parser.add_argument("--device", type=str, default=None,
                        help="设备序列号（默认自动检测）")
    parser.add_argument("--phone", type=str, default=None,
                        help=f"手机号（默认 {DEFAULT_PHONE}）")
    parser.add_argument("--cookie-file", type=str, default=DEFAULT_COOKIE_FILE,
                        help=f"Cookie 保存文件（默认 {DEFAULT_COOKIE_FILE}）")
    parser.add_argument("--skip-login", action="store_true",
                        help="跳过登录（使用已保存的 cookies）")
    parser.add_argument("--frida", action="store_true",
                        help="使用 Frida 注入开启 WebView 调试（需 root + frida-server）")
    parser.add_argument("--ocr", type=str, default="auto",
                        choices=["auto", "paddleocr", "rapidocr", "tesseract"],
                        help="OCR 引擎（默认 auto 自动检测）")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="输出详细日志")
    args = parser.parse_args()

    phone = args.phone or DEFAULT_PHONE
    cookie_file = args.cookie_file
    verbose = args.verbose

    print("╔════════════════════════════════════════════════╗")
    print("║  大麦网 抢票预约自动化 (OCR 版)                ║")
    print("║  登录 → 抢票预约 → OCR 点击已预约 → 场次票档   ║")
    print("╚════════════════════════════════════════════════╝")
    print()

    # ── Step -1: Frida 注入（可选，开启 WebView 调试） ────────────────
    frida_hook = None
    if args.frida:
        try:
            from frida_webview_debug import FridaWebViewDebugHook
            print("💉 正在通过 Frida 注入 WebView 调试 hook…")
            frida_hook = FridaWebViewDebugHook(verbose=verbose)
            if not frida_hook.attach(DAMAI_PACKAGE):
                print("⚠️ Frida attach 失败，将尝试继续（WebView 可能无法连接）")
            else:
                print("✅ Frida hook 已注入，WebView 调试已开启")
                time.sleep(2)
        except ImportError:
            print("⚠️ 未安装 frida，跳过 Frida 注入。")
            print("  安装方法：pip install frida frida-tools")
        except Exception as e:
            print(f"⚠️ Frida 注入失败：{e}")
            print("  将尝试继续，但 WebView 可能无法连接。")

    # ── Step 0: 连接设备 ─────────────────────────────────────────────
    automation = DamaiReserveOCRAutomation(
        device_serial=args.device,
        verbose=verbose,
        ocr_engine=args.ocr,
    )
    automation.connect_device()

    # ── Step 1: 启动 APP ─────────────────────────────────────────────
    automation.launch_damai_app()

    # ── Step 2: 登录 ─────────────────────────────────────────────────
    if args.skip_login:
        if try_load_cookies(automation, cookie_file):
            print("✅ 跳过登录，使用已保存的 cookies。")
        else:
            print("⚠️ 未找到已保存的 cookies，需要重新登录。")
            args.skip_login = False

    if not args.skip_login:
        if automation.check_login_status():
            print("✅ 已登录，跳过登录流程。")
        else:
            print(f"📱 手机号: {phone}")
            print()

            if not automation.login_with_terms(phone):
                print("\n❌ 登录失败，无法继续。")
                print("提示：")
                print("  1. 确认手机号正确且能接收短信")
                print("  2. 确认大麦 APP 中登录页面正常显示")
                print("  3. 如触发图形验证码，请在手机上手动完成验证")
                print("  4. 确认设备 WebView 调试已开启")
                automation.cleanup()
                sys.exit(1)

            cookies = automation.get_cookies()
            if cookies:
                automation.save_cookies(cookie_file)
            else:
                print("⚠️ 未能提取 cookies，将尝试继续…")

    # ── Step 3: 导航到抢票预约 ───────────────────────────────────────
    print()
    if not automation.navigate_to_reserve():
        print("\n❌ 无法进入「抢票预约」页面，无法继续。")
        automation.cleanup()
        sys.exit(1)

    # ── Step 4: 找到第一条已预约演出 ─────────────────────────────────
    print()
    first_show = automation.get_first_reserved_show()
    if not first_show:
        print("\n❌ 未找到已预约的演出。")
        print("提示：")
        print("  1. 确认「抢票预约」中有预约记录")
        print("  2. 确认预约未过期/取消")
        automation.cleanup()
        sys.exit(1)

    show_name = first_show.get("name", "(未知)")
    print(f"\n📌 第一条预约演出：")
    print(f"   🎭 {show_name}")
    if first_show.get("item_id"):
        print(f"   🆔 ID: {first_show['item_id']}")
    if first_show.get("url"):
        print(f"   🔗 {first_show['url']}")

    # ── Step 5: 通过 OCR 点击「已预约」查看场次和票档 ───────────────
    print()
    print("🔍 使用 OCR 模式进入预约详情…")
    if not automation.click_reserved_button():
        print("\n❌ 无法通过 OCR 点击「已预约」按钮。")
        print("将尝试提取当前页面信息…")

    # 提取场次和票档信息（OCR 版）
    time.sleep(2)  # 等待页面加载
    info = automation.extract_sessions_and_tickets()

    # 输出结果
    print()
    print(DamaiReserveAutomation.format_sessions_and_tickets(info))

    # ── 清理 ─────────────────────────────────────────────────────────
    automation.cleanup()
    if frida_hook and frida_hook.is_attached():
        frida_hook.detach()
        print("✅ Frida hook 已断开")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")

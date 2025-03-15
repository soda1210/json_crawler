#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Facebook 廣告爬蟲執行腳本
"""

import os
import sys
import argparse

# 將專案根目錄加入 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scrapers.ad_scraper import AdScraper

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='Facebook 廣告爬蟲工具')
    parser.add_argument('--json-dir', type=str, default='data/json',
                        help='JSON 檔案目錄 (預設: data/json)')
    parser.add_argument('--output-dir', type=str, default='data/images',
                        help='圖片輸出目錄 (預設: data/images)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Facebook 廣告爬蟲工具")
    print("=" * 50)
    print(f"JSON 檔案目錄: {args.json_dir}")
    print(f"圖片輸出目錄: {args.output_dir}")
    print("-" * 50)
    
    # 建立並執行爬蟲
    scraper = AdScraper(json_dir=args.json_dir, output_dir=args.output_dir)
    success = scraper.run()
    
    if success:
        print("\n✅ 爬蟲執行完成！")
    else:
        print("\n❌ 爬蟲執行失敗！")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 
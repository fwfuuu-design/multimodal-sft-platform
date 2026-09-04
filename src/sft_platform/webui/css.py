# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified in 2026 for the Multimodal Fine-tuning Platform MVP.
#
# Modified in 2026 for the multimodal fine-tuning product WebUI.

CSS = r"""
:root {
  --mvp-canvas: #f5f5f5;
  --mvp-paper: #ffffff;
  --mvp-surface-alt: #fafafa;
  --mvp-ink: #0a0a0a;
  --mvp-ink-soft: #171717;
  --mvp-muted: #737373;
  --mvp-hairline: #e5e5e5;
  --mvp-danger: #e7000b;
  --mvp-shadow: 0 1px 2px rgb(0 0 0 / 0.03), 0 8px 24px rgb(0 0 0 / 0.035);
}

body,
.gradio-container {
  background: var(--mvp-canvas) !important;
  color: var(--mvp-ink-soft) !important;
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  font-size: 14px !important;
}

.gradio-container {
  margin: 0 auto !important;
  max-width: 1280px !important;
  padding: 24px 24px 48px !important;
}

.product-header {
  align-items: flex-start !important;
  gap: 4px !important;
  margin: 8px 0 20px !important;
}

.product-header h1,
.product-header h3 {
  margin: 0 !important;
  text-align: left !important;
}

.product-header h1 {
  color: var(--mvp-ink) !important;
  font-size: clamp(28px, 3vw, 36px) !important;
  font-weight: 600 !important;
  letter-spacing: -0.035em !important;
  line-height: 1.15 !important;
}

.product-header h3 {
  color: var(--mvp-muted) !important;
  font-size: 14px !important;
  font-weight: 400 !important;
  line-height: 1.5 !important;
}

.surface-card {
  background: var(--mvp-paper) !important;
  border: 1px solid var(--mvp-hairline) !important;
  border-radius: 24px !important;
  box-shadow: var(--mvp-shadow) !important;
  padding: 20px !important;
}

.model-config-card {
  margin-bottom: 16px !important;
}

.product-tabs > .tab-nav {
  background: var(--mvp-surface-alt) !important;
  border: 1px solid var(--mvp-hairline) !important;
  border-radius: 999px !important;
  gap: 4px !important;
  margin: 0 0 12px !important;
  padding: 4px !important;
  width: fit-content !important;
}

.product-tabs > .tab-nav button {
  border: 0 !important;
  border-radius: 999px !important;
  color: var(--mvp-muted) !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  min-height: 36px !important;
  padding: 0 16px !important;
}

.product-tabs > .tab-nav button.selected {
  background: var(--mvp-ink) !important;
  color: var(--mvp-paper) !important;
}

button.primary {
  background: var(--mvp-ink) !important;
  border-color: var(--mvp-ink) !important;
  color: var(--mvp-paper) !important;
}

button.stop {
  background: transparent !important;
  border-color: var(--mvp-hairline) !important;
  color: var(--mvp-danger) !important;
}

button,
input,
textarea,
select,
.wrap {
  border-radius: 18px !important;
}

button {
  font-size: 13px !important;
  font-weight: 500 !important;
}

input,
textarea {
  background: var(--mvp-canvas) !important;
  border-color: transparent !important;
}

input:focus,
textarea:focus {
  border-color: var(--mvp-hairline) !important;
  box-shadow: 0 0 0 1px var(--mvp-ink) !important;
}

.accordion {
  background: var(--mvp-surface-alt) !important;
  border: 1px solid var(--mvp-hairline) !important;
  border-radius: 10px !important;
}

.thinking-summary {
  padding: 8px !important;
}

.thinking-summary span {
  background: var(--mvp-canvas) !important;
  border-radius: 6px !important;
  cursor: pointer !important;
  font-size: 14px !important;
  padding: 4px !important;
}

.thinking-container {
  border-left: 2px solid var(--mvp-hairline) !important;
  margin: 4px 0 !important;
  padding-left: 10px !important;
}

.thinking-container p {
  color: var(--mvp-muted) !important;
}

.modal-box {
  background: var(--mvp-paper) !important;
  border: 1px solid var(--mvp-hairline) !important;
  border-radius: 24px !important;
  box-shadow: 0 24px 80px rgb(0 0 0 / 0.16) !important;
  flex-wrap: nowrap !important;
  left: 50%;
  max-height: 750px;
  max-width: 1000px;
  overflow-y: auto;
  padding: 20px;
  position: fixed !important;
  top: 50%;
  transform: translate(-50%, -50%);
  z-index: 1000;
}

@media (max-width: 800px) {
  .gradio-container {
    padding: 16px !important;
  }

  .surface-card {
    border-radius: 18px !important;
    padding: 16px !important;
  }
}
"""

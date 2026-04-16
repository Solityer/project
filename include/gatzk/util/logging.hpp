#pragma once
// 本文件提供 artifact 导出与控制台日志的最小工具集，
// 服务 prove / verify / benchmark 输出，但不参与协议语义。

#include <map>
#include <string>
#include <vector>

namespace gatzk::util {

// 确保 artifact 目录存在。
void ensure_directory(const std::string& path);
// 逐行写出文本文件，如 commitment 顺序和 proof block 顺序。
void write_lines(const std::string& path, const std::vector<std::string>& lines);
// 写出简单 `key = value` 形式的文本元数据。
void write_key_values(const std::string& path, const std::map<std::string, std::string>& entries);
// 写出 manifest 用的扁平 JSON 对象。
void write_json_object(const std::string& path, const std::map<std::string, std::string>& entries);
// 输出运行期信息日志，辅助 formal suite 与 benchmark 诊断。
void info(const std::string& message);

}  // namespace gatzk::util

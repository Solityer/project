#include "gatzk/data/loader.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace gatzk::data {
namespace {

using algebra::FieldElement;

std::unordered_map<std::string, std::string> read_meta(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open meta file: " + path.string());
    }
    std::unordered_map<std::string, std::string> out;
    std::string line;
    while (std::getline(input, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        auto key = line.substr(0, eq);
        auto value = line.substr(eq + 1);
        key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
        value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), value.end());
        out[key] = value;
    }
    return out;
}

FieldElement quantize(double value) {
    return FieldElement::from_signed(static_cast<std::int64_t>(value >= 0.0 ? value * 16.0 + 0.5 : value * 16.0 - 0.5));
}

GraphDataset load_cached_dataset(const std::filesystem::path& root) {
    const auto meta = read_meta(root / "meta.cfg");
    GraphDataset dataset;
    dataset.name = meta.at("name");
    dataset.num_nodes = std::stoull(meta.at("num_nodes"));
    dataset.num_features = std::stoull(meta.at("num_features"));
    dataset.num_classes = std::stoull(meta.at("num_classes"));
    dataset.features.assign(dataset.num_nodes, std::vector<FieldElement>(dataset.num_features, FieldElement::zero()));

    {
        std::ifstream input(root / "features.txt");
        if (!input) {
            throw std::runtime_error("failed to open features file");
        }
        std::size_t rows = 0;
        std::size_t cols = 0;
        input >> rows >> cols;
        std::string line;
        std::getline(input, line);
        if (rows != dataset.num_nodes || cols != dataset.num_features) {
            throw std::runtime_error("feature metadata mismatch");
        }
        for (std::size_t row = 0; row < rows; ++row) {
            std::getline(input, line);
            std::istringstream iss(line);
            std::string entry;
            while (iss >> entry) {
                const auto colon = entry.find(':');
                const auto index = static_cast<std::size_t>(std::stoull(entry.substr(0, colon)));
                const auto value = std::stod(entry.substr(colon + 1));
                dataset.features[row][index] = quantize(value);
            }
        }
    }

    {
        std::ifstream input(root / "labels.txt");
        if (!input) {
            throw std::runtime_error("failed to open labels file");
        }
        dataset.labels.reserve(dataset.num_nodes);
        int label = 0;
        while (input >> label) {
            dataset.labels.push_back(label);
        }
        if (dataset.labels.size() != dataset.num_nodes) {
            throw std::runtime_error("label count mismatch");
        }
    }

    {
        std::ifstream input(root / "edges.txt");
        if (!input) {
            throw std::runtime_error("failed to open edges file");
        }
        Edge edge;
        while (input >> edge.src >> edge.dst) {
            dataset.edges.push_back(edge);
        }
    }
    return dataset;
}

void maybe_prepare_planetoid_cache(const util::AppConfig& config, const std::filesystem::path& cache_root) {
    if (!config.auto_prepare_dataset || std::filesystem::exists(cache_root / "meta.cfg")) {
        return;
    }
    const std::filesystem::path project_root(config.project_root);
    const auto python = std::filesystem::exists(project_root / ".venv/bin/python")
        ? (project_root / ".venv/bin/python").string()
        : std::string("python3");
    const auto script = (project_root / "scripts/prepare_planetoid.py").string();
    const auto command = python + " " + script +
        " --data-root " + (project_root / config.data_root).string() +
        " --dataset " + config.dataset +
        " --cache-root " + (project_root / config.cache_root).string();
    if (std::system(command.c_str()) != 0) {
        throw std::runtime_error("failed to prepare planetoid cache for dataset " + config.dataset);
    }
}

}  // namespace

GraphDataset load_dataset(const util::AppConfig& config) {
    const std::filesystem::path project_root(config.project_root);
    if (config.dataset == "toy") {
        return load_cached_dataset(project_root / "data/toy");
    }

    const auto cache_root = project_root / config.cache_root / config.dataset;
    maybe_prepare_planetoid_cache(config, cache_root);
    return load_cached_dataset(cache_root);
}

LocalGraph extract_local_subgraph(const GraphDataset& dataset, std::size_t center_node, std::size_t local_nodes) {
    if (dataset.num_nodes == 0) {
        throw std::runtime_error("cannot extract a subgraph from an empty dataset");
    }

    const std::size_t target = std::min(local_nodes, dataset.num_nodes);
    const std::size_t start = std::min(center_node, dataset.num_nodes - 1);

    if (target == dataset.num_nodes) {
        LocalGraph local;
        local.name = dataset.name + "_full";
        local.num_nodes = dataset.num_nodes;
        local.num_features = dataset.num_features;
        local.num_classes = dataset.num_classes;
        local.absolute_ids.resize(dataset.num_nodes);
        local.features = dataset.features;
        local.labels = dataset.labels;
        for (std::size_t node = 0; node < dataset.num_nodes; ++node) {
            local.absolute_ids[node] = node;
        }

        std::unordered_set<std::uint64_t> seen;
        for (const auto& edge : dataset.edges) {
            const auto key = (static_cast<std::uint64_t>(edge.src) << 32U) | static_cast<std::uint64_t>(edge.dst);
            if (seen.insert(key).second) {
                local.edges.push_back(edge);
            }
        }
        for (std::size_t node = 0; node < dataset.num_nodes; ++node) {
            const auto key = (static_cast<std::uint64_t>(node) << 32U) | static_cast<std::uint64_t>(node);
            if (seen.insert(key).second) {
                local.edges.push_back(Edge{node, node});
            }
        }
        std::stable_sort(local.edges.begin(), local.edges.end(), [](const Edge& lhs, const Edge& rhs) {
            return lhs.dst < rhs.dst;
        });
        return local;
    }

    std::vector<std::vector<std::size_t>> adjacency(dataset.num_nodes);
    for (const auto& edge : dataset.edges) {
        adjacency[edge.src].push_back(edge.dst);
        adjacency[edge.dst].push_back(edge.src);
    }

    std::vector<std::size_t> selected;
    std::vector<bool> visited(dataset.num_nodes, false);
    std::queue<std::size_t> queue;
    queue.push(start);
    visited[start] = true;

    while (!queue.empty() && selected.size() < target) {
        const auto node = queue.front();
        queue.pop();
        selected.push_back(node);
        for (const auto neighbor : adjacency[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
    }

    for (std::size_t node = 0; selected.size() < target && node < dataset.num_nodes; ++node) {
        if (!visited[node]) {
            visited[node] = true;
            selected.push_back(node);
        }
    }

    std::unordered_map<std::size_t, std::size_t> global_to_local;
    for (std::size_t i = 0; i < selected.size(); ++i) {
        global_to_local[selected[i]] = i;
    }

    LocalGraph local;
    local.name = dataset.name + "_local";
    local.num_nodes = selected.size();
    local.num_features = dataset.num_features;
    local.num_classes = dataset.num_classes;
    local.absolute_ids = selected;
    local.features.reserve(local.num_nodes);
    local.labels.reserve(local.num_nodes);
    for (const auto absolute_id : selected) {
        local.features.push_back(dataset.features[absolute_id]);
        local.labels.push_back(dataset.labels[absolute_id]);
    }

    std::unordered_set<std::uint64_t> seen;
    for (const auto& edge : dataset.edges) {
        if (!global_to_local.contains(edge.src) || !global_to_local.contains(edge.dst)) {
            continue;
        }
        const Edge local_edge{global_to_local[edge.src], global_to_local[edge.dst]};
        const auto key = (static_cast<std::uint64_t>(local_edge.src) << 32U) | static_cast<std::uint64_t>(local_edge.dst);
        if (seen.insert(key).second) {
            local.edges.push_back(local_edge);
        }
    }

    for (std::size_t node = 0; node < local.num_nodes; ++node) {
        const auto key = (static_cast<std::uint64_t>(node) << 32U) | static_cast<std::uint64_t>(node);
        if (seen.insert(key).second) {
            local.edges.push_back(Edge{node, node});
        }
    }

    std::stable_sort(local.edges.begin(), local.edges.end(), [](const Edge& lhs, const Edge& rhs) {
        return lhs.dst < rhs.dst;
    });

    return local;
}

}  // namespace gatzk::data

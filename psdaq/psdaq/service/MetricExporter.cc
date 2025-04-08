#include <iostream>
#include <map>
#include <algorithm>    // std::find_if
#include <sys/stat.h>
#include "MetricExporter.hh"
#include "psalg/utils/SysLog.hh"

static const unsigned PROM_PORT_BASE = 9200;       // Prometheus montitoring port
static const unsigned MAX_PROM_PORTS = 100;

using logging = psalg::SysLog;

std::unique_ptr<prometheus::Exposer>
Pds::createExposer(const std::string& prometheusDir,
                   const std::string& hostname)
{
    std::unique_ptr<prometheus::Exposer> exposer;

    // Find and register a port to use with Prometheus for run-time monitoring
    unsigned port = 0;
    for (unsigned i = 0; i < MAX_PROM_PORTS; ++i) {
        try {
            port = PROM_PORT_BASE + i;
            std::string fileName = prometheusDir + "/drpmon_" + hostname + "_" + std::to_string(i) + ".yaml";
            // Commented out the existing file check so that the file's date is refreshed
            //struct stat buf;
            //if (stat(fileName.c_str(), &buf) != 0) {
                FILE* file = fopen(fileName.c_str(), "w");
                if (file) {
                    logging::debug("Writing %s\n", fileName.c_str());
                    fprintf(file, "- targets:\n    - '%s:%d'\n", hostname.c_str(), port);
                    fclose(file);
                } else {
                    // %m will be replaced by the string strerror(errno)
                    logging::debug("Error creating file %s: %m", fileName.c_str());
                }
            //} else {
            //    // File already exists; no need to rewrite it
            //    // @todo: Perhaps 'touch' the file to refresh its date here
            //    // so that we can see which ones are old and likely stale?
            //}

            exposer = std::make_unique<prometheus::Exposer>("0.0.0.0:"+std::to_string(port), 1);
            break;
        }
        catch(const std::runtime_error& e) {
            logging::debug("Could not start run-time monitoring server on port %d:\n  %s",
                           port, e.what());
        }
    }

    if (exposer) {
        logging::info("Prometheus run-time monitoring data is on port %d", port);
    } else {
        logging::warning("No port found for prometheus run-time monitoring data");
    }

    return exposer;
}


static
prometheus::MetricFamily makeMetric(const std::string& name,
                                    const std::map<std::string, std::string>& labels,
                                    prometheus::MetricType type)
{
    prometheus::MetricFamily family;
    family.name = name;
    family.type = type;

    prometheus::ClientMetric metric;
    for (const auto& labelPair : labels) {
        auto label = prometheus::ClientMetric::Label{};
        label.name = labelPair.first;
        label.value = labelPair.second;
        metric.label.emplace_back(label);

    }
    family.metric.emplace_back(metric);
    return family;
}

int Pds::MetricExporter::find(const std::string& name) const
{
    int i = 0;

    for (auto family : m_families) {
        if (family.name == name) {
            return i;
        }
        ++i;
    }
    return -1;
}

void Pds::MetricExporter::_erase(unsigned index)
{
    m_families.erase(m_families.begin() + index);
    m_values.  erase(m_values.  begin() + index);
    m_floats.  erase(m_floats.  begin() + index);
    m_histos.  erase(m_histos.  begin() + index);
    m_type.    erase(m_type.    begin() + index);
    m_previous.erase(m_previous.begin() + index);
}

void Pds::MetricExporter::add(const std::string& name,
                              const std::map<std::string, std::string>& labels,
                              Pds::MetricType type, std::function<int64_t()> value)
{
    // Avoid Collect() from processing incomplete entries
    std::lock_guard<std::mutex> lock(m_mutex);
    int i = find(name);
    if (i != -1) {
        _erase(i);
    }
    prometheus::MetricType prometheusType;
    switch (type) {
        case Pds::MetricType::Gauge:
            prometheusType = prometheus::MetricType::Gauge;
            break;
        case Pds::MetricType::Constant:
        case Pds::MetricType::Counter:
            prometheusType = prometheus::MetricType::Counter;
            break;
        case Pds::MetricType::Rate:
            prometheusType = prometheus::MetricType::Gauge;
            break;
        default:
          std::cout<<"Bad type: "<<int(type)<<"\n";
    }
    m_families.emplace_back(makeMetric(name, labels, prometheusType));
    m_values.push_back(value);
    m_floats.push_back(nullptr);
    m_histos.push_back(nullptr);        // Placeholder; not used
    m_type.push_back(type);
    Pds::Previous previous;
    if (type == Pds::MetricType::Rate) {
        previous.value = value();
        previous.time = std::chrono::steady_clock::now();
    }
    m_previous.push_back(previous);
}

void Pds::MetricExporter::addFloat(const std::string& name,
                                   const std::map<std::string, std::string>& labels,
                                   std::function<bool(double&)> value)
{
    // Avoid Collect() from processing incomplete entries
    std::lock_guard<std::mutex> lock(m_mutex);
    int i = find(name);
    if (i != -1) {
        _erase(i);
    }
    prometheus::MetricType prometheusType = prometheus::MetricType::Gauge;
    m_families.emplace_back(makeMetric(name, labels, prometheusType));
    m_values.push_back(nullptr);
    m_floats.push_back(value);
    m_histos.push_back(nullptr);        // Placeholder; not used
    m_type.push_back(Pds::MetricType::Float);
    Pds::Previous previous;
    m_previous.push_back(previous);
}

void Pds::MetricExporter::constant(const std::string& name,
                                   const std::map<std::string, std::string>& labels,
                                   int64_t constant)
{
    Pds::MetricType type = Pds::MetricType::Constant;
    std::function<int64_t()> value;    // Placeholder; not used

    add(name, labels, type, value);

    m_families.back().metric[0].counter.value = static_cast<double>(constant);
}

std::vector<prometheus::MetricFamily> Pds::MetricExporter::Collect() const
{
    std::lock_guard<std::mutex> lock(m_mutex);

    // std::cout<<"Collect()\n";
    for (size_t i=0; i<m_type.size(); i++) {
        //std::cout<<"Collector  "<<m_families[i].name<<'\n';
        switch (m_type[i]) {
            case Pds::MetricType::Rate: {
                int64_t newValue = m_values[i]();
                auto now = std::chrono::steady_clock::now();
                int64_t previousValue = m_previous[i].value;
                int64_t difference = 0UL;
                if (newValue > previousValue) {
                    difference = newValue - previousValue;
                }
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - m_previous[i].time).count();
                double rate = static_cast<double>(difference) / static_cast<double>(duration) * 1.0e6;
                setValue(m_families[i], rate);
                m_previous[i].value = newValue;
                m_previous[i].time = now;
                break;
            }
            case Pds::MetricType::Histogram: {
                m_histos[i]->collect(m_families[i]);
                break;
            }
            case Pds::MetricType::Constant: {
                break;                    // Nothing to do
            }
            case Pds::MetricType::Float: {
                double value;
                if (m_floats[i](value)) { // Update only when valid
                    setValue(m_families[i], value);
                }
                break;
            }
            default: {
                setValue(m_families[i], m_values[i]());
                break;
            }
        }
    }
    return m_families;
}


Pds::PromHistogram::PromHistogram(unsigned numBins, double binWidth, double binMin) :
  m_boundaries(numBins),
  m_counts    (numBins + 1),
  m_sum       (0.0)
{
    //if (numBins)
    //    printf("PromHisto: numBins %u, binWidth %f, binMin %f\n", numBins, binWidth, binMin);
    for (unsigned i = 0; i < m_boundaries.size(); ++i) {
        m_boundaries[i] = binMin + binWidth * static_cast<double>(i);
    }
}

std::shared_ptr<Pds::PromHistogram>
    Pds::MetricExporter::histogram(const std::string& name,
                                   const std::map<std::string, std::string>& labels,
                                   unsigned numBins, double binWidth, double binMin)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    int i = find(name);
    if (i != -1) {
        _erase(i);
    }
    Pds::MetricType type = Pds::MetricType::Histogram;
    prometheus::MetricType prometheusType = prometheus::MetricType::Histogram;
    m_families.emplace_back(makeMetric(name, labels, prometheusType));
    m_families.back().metric[0].histogram.bucket.resize(numBins + 1);
    m_values.push_back(nullptr);        // Placeholder; won't be called
    m_histos.push_back(std::make_shared<PromHistogram>(numBins, binWidth, binMin));
    m_type.push_back(type);
    m_previous.push_back({});           // Placeholder: unused

    //for (unsigned i = 0; i < m_type.size(); ++i)
    //  printf("type[%u] %d\n", i, int(m_type[i]));
    return m_histos.back();
}

void Pds::PromHistogram::collect(prometheus::MetricFamily& family)
{
    auto& metric = family.metric[0];

    auto cumulative_count = 0ULL;
    for (std::size_t i = 0; i < m_counts.size(); i++) {
        cumulative_count += m_counts[i];
        auto& bucket = metric.histogram.bucket[i];
        bucket.cumulative_count = cumulative_count;
        bucket.upper_bound = (i == m_boundaries.size()
                           ? std::numeric_limits<double>::infinity()
                           : m_boundaries[i]);
        //printf("collect: i %zd, cnt %lu, ub %f\n", i, bucket.cumulative_count, bucket.upper_bound);
    }
    metric.histogram.sample_count = cumulative_count;
    metric.histogram.sample_sum = m_sum;

    //printf("collect: count %llu, sum %f\n", cumulative_count, m_sum);
    //for (unsigned i = 0; i < metric.histogram.bucket.size(); ++i)
    //    printf("collect: bucket[%u] cnt %lu, ub %f\n",
    //           i, metric.histogram.bucket[i].cumulative_count, metric.histogram.bucket[i].upper_bound);
}

void Pds::PromHistogram::observe(double sample)
{
    auto bin = static_cast<std::size_t>(std::distance(
        m_boundaries.begin(),
        std::find_if(m_boundaries.begin(), m_boundaries.end(),
                     [sample](double boundary) { return boundary >= sample; })));
    if (bin >= 0 && bin < m_counts.size()) {
        m_counts[bin]++;
        m_sum += sample;
    }
    else {
      fprintf(stderr, "%s:\n  Bin number %zd is out of range [0, %zd) for sample %f ([%f - %f))\n",
              __PRETTY_FUNCTION__, bin, m_counts.size(), sample, *m_boundaries.begin(), *m_boundaries.end());
    }
    //printf("observe: sample %f, bin %zd, counts %lu, sum %f\n",
    //       sample, bin, m_counts[bin], m_sum);
}

void Pds::PromHistogram::clear()
{
    for (unsigned i = 0; i < m_counts.size(); ++i)
        m_counts[i] = 0;
    m_sum = 0.0;
}

# Vendored parser

The optional native swarm worker uses nlohmann/json 3.11.3, licensed under MIT.

- Header: https://github.com/nlohmann/json/blob/v3.11.3/single_include/nlohmann/json.hpp
- License: `nlohmann/LICENSE.MIT`
- Header SHA-256: `9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6`

The numerical core and Python extension do not include or link this parser unless the worker target is built. libcurl and OpenSSL are system dependencies of the worker; they are not vendored here.

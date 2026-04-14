{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      eachSystem =
        f:
        nixpkgs.lib.genAttrs nixpkgs.lib.systems.flakeExposed (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = eachSystem (pkgs: {
        default =
          let
            pg = pkgs.postgresql_16.withPackages (ps: [ ps.pgvector ]);
          in
          pkgs.mkShell {
            packages = [
              pkgs.python312
              pkgs.uv
              pkgs.ruff
              pkgs.jq
              pg
            ];

            PGDATA = "$PWD/.pgdata";
            PGHOST = "$PWD/.pgdata";
            PGDATABASE = "vex_brain";

            shellHook = ''
              export PGDATA="$PWD/.pgdata"
              export PGHOST="$PWD/.pgdata"
              export PGDATABASE="vex_brain"
              export VEX_BRAIN_DATABASE_URL="postgresql:///vex_brain?host=$PWD/.pgdata"

              # Helper: start local postgres
              pg-start() {
                if [ ! -d "$PGDATA" ]; then
                  echo "Initialising Postgres data directory..."
                  initdb --no-locale --encoding=UTF8 -D "$PGDATA"
                  echo "unix_socket_directories = '$PGDATA'" >> "$PGDATA/postgresql.conf"
                  echo "listen_addresses = '''" >> "$PGDATA/postgresql.conf"
                fi
                if ! pg_isready -q 2>/dev/null; then
                  echo "Starting Postgres..."
                  pg_ctl -D "$PGDATA" -l "$PGDATA/postgres.log" start
                  sleep 1
                  createdb vex_brain 2>/dev/null || true
                  echo "Postgres ready — vex_brain database"
                else
                  echo "Postgres already running"
                fi
              }

              # Helper: stop local postgres
              pg-stop() {
                pg_ctl -D "$PGDATA" stop 2>/dev/null || echo "Postgres not running"
              }

              echo "vex-brain dev shell"
              echo "  pg-start  — start local Postgres with pgvector"
              echo "  pg-stop   — stop local Postgres"
            '';
          };
      });
    };
}
